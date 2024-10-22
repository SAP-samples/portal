from __future__ import annotations

import datetime
import logging
import pickle
import warnings
from abc import ABC, abstractmethod
from time import sleep
from typing import Any, Collection, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import zmq
from transformers import AutoTokenizer

from portal.constants import DEFAULT_ZMQ_PORT
from portal.data.data_utils import encode_dates, encode_numbers, to_numeric
from portal.model.torch_modules import embedding_model_to_dimension_and_pooling
from portal.scripts.start_embedding_server import embedding_server_starter

logger = logging.getLogger(__name__)
# The following warning should be ignored by passing `format='mixed'` to pd.to_datetime,
# but somehow doing so changes the outputs... so we'll just ignore it.
warnings.filterwarnings('ignore', message='Could not infer format, so each element will be parsed.*')


class EmbeddingServerRowEmbedder:
    """
    Specialized RowEmbedder that uses a ZMQ-connected embedding server to compute the embeddings.
    """
    def __init__(self,
                 sentence_embedding_model_name,
                 zmq_port: int = DEFAULT_ZMQ_PORT,
                 fake=False):
        self.embedding_dim = embedding_model_to_dimension_and_pooling[sentence_embedding_model_name][0]
        self.socket: Union[zmq.Socket, None] = None
        self.zmq_port = zmq_port
        self.fake = fake
        self.sentence_embedding_model_name = sentence_embedding_model_name

    def embed_columns_and_contents(self,
                                   row_provider: LazySeriesProvider,
                                   is_text: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a row of a dataframe, return two tensors:
        - column_embeddings: one embedding per column name
        - content_embeddings: one embedding per cell content
        The encoding must be the same as done in wikipedia_fixes.py
        or it won't be found in the database!
        """
        if is_text is None:
            not_date = pd.isnull(row_provider.date)
            not_number = pd.isnull(row_provider.numeric)
            is_text = (not_date & not_number).values
        row = row_provider.string
        assert isinstance(row.values, np.ndarray) and isinstance(is_text, np.ndarray)
        strings_to_embed = row.index.tolist() + row.values[is_text].tolist()
        embeddings = self.texts_to_tensor(strings_to_embed)
        column_name_embeddings, content_embeddings = embeddings[:len(row)], embeddings[len(row):]
        real_content_embeddings = torch.zeros_like(column_name_embeddings)
        real_content_embeddings[torch.tensor(is_text)] = content_embeddings
        return column_name_embeddings, real_content_embeddings

    def socket_init(self):
        self.socket = zmq.Context().socket(zmq.REQ)
        self.socket.connect(f'tcp://localhost:{self.zmq_port}')
        # Timeout after 10 seconds
        self.socket.setsockopt(zmq.RCVTIMEO, 10000)
        self.socket.setsockopt(zmq.LINGER, 0)

    def texts_to_tensor(self, texts: Collection[str]) -> torch.Tensor:
        if self.fake:
            return torch.zeros((len(texts), self.embedding_dim), dtype=torch.float16)

        if not texts:
            return torch.zeros((0, self.embedding_dim), dtype=torch.float16)
        # Make sure we're actually dealing with texts
        texts = [str(x) for x in texts]

        missing_texts = list(set(texts))
        result_dict = {}

        if missing_texts:
            serialized_data = pickle.dumps(missing_texts)
            if self.socket is None:
                self.socket_init()
            assert self.socket is not None, 'Socket not initialized!'
            self.socket.send(serialized_data)
            try:
                response = self.socket.recv()
                response = pickle.loads(response)
            except zmq.error.Again as e:
                print(f'Warning: no response from server ({e}).')
                print('You might have forgotten to start it, or it might have been killed?')
                print('Trying restarting it.')
                embedding_server_starter.restart()
                sleep(30)
                del self.socket
                self.socket_init()
                assert isinstance(self.socket, zmq.Socket)
                self.socket.send(serialized_data)
                response = self.socket.recv()
                response = pickle.loads(response)

            for i, text in enumerate(missing_texts):
                result_dict[text] = response[i]

        results = [result_dict[text] for text in texts]
        result = b''.join(results)
        return torch.frombuffer(result, dtype=torch.float16).view(len(texts), self.embedding_dim)


class LazySeriesProvider:
    """
    Class that provides a lazy interface to a pandas Series, allowing to extract
    different types of data from it (numeric or datetime).
    Works for both dataframe rows or columns.
    """
    def __init__(self, row: pd.Series, df_dtypes: Union[Tuple[Any], None] = None):
        self._row = row
        self._string_row = None
        self._numeric_row = None
        self._date_row = None
        self._time_row = None
        self.df_dtypes = df_dtypes

    @property
    def original(self):
        return self._row

    @property
    def string(self) -> pd.Series:
        if self._string_row is None:
            self._string_row = self._row.apply(str)
        return self._string_row

    @property
    def numeric(self) -> pd.Series:
        if self._numeric_row is None:
            self._numeric_row = to_numeric(self._row, self.date, self.time)
        return self._numeric_row

    def update_time_and_date(self):
        # TODO: check if this UTC=True causes problems in some cases.
        # It is needed sometimes to convert otherwise it raises:
        # Tz-aware datetime.datetime cannot be converted to datetime64 unless utc=True
        date_and_time = pd.to_datetime(self._row.apply(str), errors='coerce', utc=True)
        # date_and_time[:] = pd.NaT
        self._time_row = date_and_time.dt.time
        self._date_row = date_and_time.dt.date.copy()

        # pd.to_datetime has an incosistency:
        # - if input is a string like '11:32', it will detect "today at 11:32"
        # - if input is a datetime.time object it will convert to NaT
        # We want NaT in that case.
        # So if the input type was really string, we can only discard any "today" result
        # Otherwise, we check the type and use the detected date only if it was
        # a datetime.datetime or datetime.date object, but not a datetime.time object

        indices_to_check = self._date_row.index[self._date_row == datetime.datetime.now().date()]

        for index in indices_to_check:
            if not isinstance(self._row[index], datetime.datetime) and not isinstance(self._row[index], datetime.date):
                self._date_row[index] = pd.NaT

    @property
    def time(self) -> pd.Series:
        if self._time_row is None:
            self.update_time_and_date()
            assert self._time_row is not None, 'Time row not initialized'

        return self._time_row

    @property
    def date(self) -> pd.Series:
        if self._date_row is None:
            self.update_time_and_date()
            assert self._date_row is not None, 'Date row not initialized'

        return self._date_row


class SpecialTokenizer(ABC):
    @abstractmethod
    def encode(self, row_provider: LazySeriesProvider) -> Dict[str, torch.Tensor]:
        ...


class DateTokenizer(SpecialTokenizer):
    def encode(self, row_provider: LazySeriesProvider) -> Dict[str, torch.Tensor]:
        row = row_provider.date
        date_year, date_month, date_day, date_weekday, date_holidays = encode_dates(row)
        is_date: 'pd.Series[bool]' = ~pd.isnull(row)
        return {
            # The first block has dimension 1, simply one per column
            'is_date': torch.tensor(is_date.values),
            'date_year': torch.tensor(date_year),
            'date_month': torch.tensor(date_month),
            'date_day': torch.tensor(date_day),
            'date_weekday': torch.tensor(date_weekday),
            # date_holidays has dimension 2, each column has 120 values (0 or 1, if it's a holiday in a given country)
            'date_holidays': torch.tensor(date_holidays, dtype=torch.uint8),
        }


class NumberTokenizer(SpecialTokenizer):
    def encode(self, row_provider: LazySeriesProvider) -> Dict[str, torch.Tensor]:
        row = row_provider.numeric
        is_positive, exponent, fraction_bin, delta = encode_numbers(row)
        is_date: 'pd.Series[bool]' = ~pd.isnull(row_provider.date)
        is_number = ~is_date & ~pd.isnull(row)
        return {
            'is_number': torch.tensor(is_number.values),
            'is_positive': torch.tensor(is_positive, dtype=torch.int32),
            'exponent': torch.tensor(exponent),
            'fraction_bin': torch.tensor(fraction_bin),
            'delta': torch.tensor(delta, dtype=torch.float32),
        }


class StringTokenizer(SpecialTokenizer):
    def __init__(self,
                 sentence_embedding_model_name: str,
                 zmq_port: int = DEFAULT_ZMQ_PORT,
                 fake=False):
        self._tokenizer = AutoTokenizer.from_pretrained(sentence_embedding_model_name)
        self.embedding_dim = embedding_model_to_dimension_and_pooling[sentence_embedding_model_name][0]
        self._row_embedder = EmbeddingServerRowEmbedder(zmq_port=zmq_port,
                                                        fake=fake,
                                                        sentence_embedding_model_name=sentence_embedding_model_name)

    @property
    def pad_token_id(self):
        return self._tokenizer.pad_token_id

    def cleanup_socket(self):
        del self._row_embedder.socket
        self._row_embedder.socket = None

    def texts_to_tensor(self, texts: Collection[str]) -> torch.Tensor:
        return self._row_embedder.texts_to_tensor(texts)

    def encode(self, row_provider: LazySeriesProvider, is_text: Optional[np.ndarray] = None) -> Dict[str, torch.Tensor]:
        column_embeddings, content_embeddings = self._row_embedder.embed_columns_and_contents(row_provider, is_text)
        return {
            # column_embeddings and content_embeddings have 2 dim, shape = (num_columns, embedding_size)
            # e.g. embedding_size for all-MiniLM-L6-v2 is 384
            'column_embeddings': column_embeddings,
            'content_embeddings': content_embeddings
        }


class ModularizedRowTokenizer:
    def __init__(self,
                 sentence_embedding_model_name: str,
                 zmq_port=DEFAULT_ZMQ_PORT,
                 use_number_percentiles=False,
                 fake_string_embeddings=False):
        self.string_tokenizer = StringTokenizer(sentence_embedding_model_name,
                                                zmq_port=zmq_port,
                                                fake=fake_string_embeddings)
        self.date_tokenizer = DateTokenizer()
        self.number_tokenizer = NumberTokenizer()
        if use_number_percentiles:
            raise NotImplementedError('Percentiles not yet implemented (or not anymore?)')

    @property
    def pad_token_id(self):
        return self.string_tokenizer.pad_token_id

    def __call__(self,
                 row: Union[pd.Series, LazySeriesProvider],
                 override_is_text: Optional[np.ndarray] = None) -> dict:
        # check if unexpected precompute rows are passed
        if isinstance(row, LazySeriesProvider):
            row_provider = row
        else:
            row_provider = LazySeriesProvider(row)
        output = self.date_tokenizer.encode(row_provider)
        output.update(self.number_tokenizer.encode(row_provider))
        is_text = ~(output['is_date'] | output['is_number']).numpy()
        if override_is_text is not None:
            is_text |= override_is_text
            output['is_date'] &= torch.tensor(~is_text)
            output['is_number'] &= torch.tensor(~is_text)
        output.update(self.string_tokenizer.encode(row_provider, is_text))
        return output
