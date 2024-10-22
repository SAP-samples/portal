import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union
import pydantic
from typing_extensions import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from portal.constants import TENANT_ID_COL, UNKNOWN_CLASS_LABEL_ID
from portal.data.data_utils import attempt_convert_dates, to_numeric, tokenized_row_encoder
from portal.data.one_token_per_cell import ModularizedRowTokenizer, mask_one_row

PORTAL_DATA_DIR = Path(__file__).absolute().parent


class BaseModelNoExtras(pydantic.BaseModel):
    """Deriving from this class will prevent (accidental) parsing of
    inputs that come with extra fields;
    this may not be what we want everywhere, but using it as a default
    can prevent not noticing silly errors (typos, ..)
    """
    model_config = pydantic.ConfigDict(extra='forbid')


class UseCaseTarget(BaseModelNoExtras):
    column: str
    task: Union[Literal['classification'], Literal['regression']]


class GenericTokenizedDataset(ABC, Dataset):
    def __init__(self,
                 data: pd.DataFrame,
                 target_fields: List[UseCaseTarget],
                 input_features: Optional[List[str]] = None,
                 drop_tenant_id_column: bool = True):
        self.target_fields = target_fields
        target_columns = [t.column for t in target_fields]

        self.current_epoch = 0
        self.drop_tenant_id_column = drop_tenant_id_column  # whether to input __tenant_id__ column to the model
        if input_features is None:
            self.df = data.drop(target_columns, axis=1)
        else:
            features_in_df = list(set(input_features).intersection(set(data.columns)))
            if len(features_in_df) != len(input_features):
                print(
                    f"{', '.join(set(input_features) - set(features_in_df))} were not found in the training data and hence will be left out. "
                )
            if target_columns is not None and set(target_columns).intersection(set(features_in_df)):
                print('\nFound target columns:')
                print(set(target_columns).intersection(set(features_in_df)))
                print('as part of input_features, removing them or they will be used to predict!\n')
                features_in_df = list(set(features_in_df) - set(target_columns))
            self.df = data[features_in_df].copy()

        # targets may not be supplied in some cases as it's optional
        if target_columns:
            available_columns = [c for c in target_columns if c in data.columns]
            if not available_columns:
                print('Warning, none of the target columns were found in the data. Target initialized to None.')
                self.target = None
            elif len(available_columns) < len(target_columns):
                print('Warning, some columns in target_columns were not found in the data. Skipping those!')
                self.target_fields = [t for t in target_fields if t.column in available_columns]
            else:
                self.target = data[available_columns].copy()
        else:
            self.target = None
            print(
                'Warning. No target column passed, be sure to add a target if you are training. Target initialized to None.'
            )

        self.convert_dates()

        #save a copy of df and targets for any resampling (downsampling/upsampling)
        self.df_copy = self.df.copy()
        if self.target is not None:
            self.target_copy = self.target.copy()

    @property
    def shape(self):
        return self.df.shape

    def get_column_names(self):
        return [c for c in self.df.columns]

    def get_constant_columns(self):
        constant_cols = [c for c in self.df.columns if c != TENANT_ID_COL and len(self.df[c].unique()) == 1]
        useful_cols = [c for c in self.df.columns if c == TENANT_ID_COL or len(self.df[c].unique()) > 1]
        return constant_cols, useful_cols

    def get_constant_targets(self):
        if not isinstance(self.target, pd.DataFrame):
            self.target = pd.DataFrame(self.target)
        target = self.target.copy()
        for target_column in self.target.columns:
            target[target_column] = target[target_column].apply(lambda value: str(value).split('::')[-1]
                                                                if '::' in str(value) else value)
        constant_targets = [c for c in self.target.columns if len(target[c].unique()) == 1]
        return constant_targets

    def drop_targets(self, targets_to_drop):
        if self.target is None:
            raise ValueError('Tried to drop targets, but no target found!')
        self.target = self.target.drop(targets_to_drop, axis=1)
        self.target_fields = [t for t in self.target_fields if t.column not in targets_to_drop]

    def select_columns(self, column_names: List[str]):
        self.df = self.df[column_names]

    def load_column_to_id(self, list_of_columns_path, df_columns):
        try:
            with open(list_of_columns_path) as fp:
                columns = json.load(fp)
        except FileNotFoundError:
            columns = df_columns.to_list()
            dump = True
        else:
            columns_set = set(columns)
            if set(df_columns).difference(columns_set):
                columns = columns + [c for c in df_columns if c not in columns_set]
                dump = True
            else:
                dump = False

        if dump:
            print('Saving updated columns to', list_of_columns_path)
            with open(list_of_columns_path, 'w') as fp:
                json.dump(columns, fp, indent=4)

        # Make sure we add unknown in position 0
        # (which we might also use for out-of-dict column names; which hopefully will never happen)
        self.columns = ['unknown'] + [c for c in columns if c != 'unknown']

        self.column_to_id = {column: idx for idx, column in enumerate(self.columns)}

    def __len__(self):
        return len(self.df)

    def get_target(self, index):
        if self.target is None or len(self.target) == 0:
            return None
        return self.target.loc[index].to_dict()

    def set_current_epoch(self, current_epoch):
        self.current_epoch = current_epoch

    def convert_dates(self):
        df = self.df
        attempt_convert_dates(df)

    @abstractmethod
    def __getitem__(self, index):
        ...

    def convert_classes_to_ids(self, id_mappings: Dict[str, int], target_column: str):
        """
        Convert class labels to IDs which the model can understand.
        :param id_mappings: dict containing the mapping between class values and the numeric IDs, per target i.e. {'class0': 0, 'class1': 1}
        """
        if self.target is None:
            print('Tried to convert target to IDs, but no target found!')
        else:
            print(f'Converting target column {target_column} classes to IDs for classification')
            self.target[target_column] = self.target[target_column].apply(
                lambda x: id_mappings.get(x, UNKNOWN_CLASS_LABEL_ID))

    def convert_target_to_float(self, target_column):
        """
        Convert target values to float, as some models (e.g. regression) require this.
        """
        if self.target is None:
            print('Tried to convert target to float, but no target found!')
        else:
            self.target[target_column] = self.target[target_column].astype(np.float64)

    def downsample_classes(self):

        # Identify classes and their counts before downsampling
        before_classes, before_counts = np.unique(self.target_copy, return_counts=True)
        print('Class distribution before downsampling:')
        for cls, count in zip(before_classes, before_counts):
            print(f'Class {cls}: {count} samples')

        # Downsampling logic
        classes, counts = np.unique(self.target_copy, return_counts=True)
        n_samples = counts.min(
        )  # Target number of samples per class for balancing, right now just naive even balancing

        new_indices = []
        for cls in classes:
            cls_indices = np.where(self.target_copy == cls)[0]
            cls_sample_indices = np.random.choice(cls_indices, n_samples, replace=False)
            new_indices.extend(cls_sample_indices)

        np.random.shuffle(new_indices)

        # Set new indices with balanced classes
        self.indices = np.array(new_indices)

        # Update DataFrame and target variable to match the new indices
        self.df = self.df_copy.iloc[self.indices].reset_index(drop=True)
        self.target = self.target_copy.iloc[self.indices].reset_index(drop=True)

        # Identify classes and their counts after downsampling
        after_classes, after_counts = np.unique(self.target, return_counts=True)
        print('Class distribution after downsampling:')
        for cls, count in zip(after_classes, after_counts):
            print(f'Class {cls}: {count} samples')

    def upsample_classes(self):
        # Identify classes and their counts before upsampling
        before_classes, before_counts = np.unique(self.target_copy, return_counts=True)
        print('Class distribution before upsampling:')
        for cls, count in zip(before_classes, before_counts):
            print(f'Class {cls}: {count} samples')

        # Upsampling logic
        max_count = before_counts.max()  # Find the count of the majority class

        new_indices = []
        for cls in before_classes:
            cls_indices = np.where(self.target_copy == cls)[0]
            # Calculate the number of samples to generate for balancing
            n_samples_needed = max_count - before_counts[before_classes == cls][0]
            if n_samples_needed > 0:  # If this class needs upsampling
                # Sample with replacement
                cls_sample_indices = np.random.choice(cls_indices, n_samples_needed, replace=True)
                cls_indices = np.concatenate((cls_indices, cls_sample_indices))
            new_indices.extend(cls_indices)

        # Shuffling the new indices to mix classes
        np.random.shuffle(new_indices)

        # Set new indices to create a balanced DataFrame
        self.indices = np.array(new_indices)

        # Update DataFrame and target variable to match the new indices
        self.df = self.df_copy.iloc[self.indices].reset_index(drop=True)
        self.target = self.target_copy.iloc[self.indices].reset_index(drop=True)

        # Identify classes and their counts after upsampling
        after_classes, after_counts = np.unique(self.target, return_counts=True)
        print('Class distribution after upsampling:')
        for cls, count in zip(after_classes, after_counts):
            print(f'Class {cls}: {count} samples')


class GenericTokenizedDatasetManyTokensPerCell(GenericTokenizedDataset):
    def __init__(self,
                 tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                 data: pd.DataFrame,
                 target_fields: List[UseCaseTarget],
                 input_features: Optional[List[str]] = None,
                 list_of_columns_path: str = f'{PORTAL_DATA_DIR}/list_of_columns.json'):

        if not tokenizer.init_kwargs.get('add_prefix_space'):
            raise ValueError('Must instantiate the tokenizer with add_prefix_space=True, otherwise '
                             'text from consecutive cells will be merged into a single word.')
        self.tokenizer = tokenizer

        super().__init__(data=data, input_features=input_features, target_fields=target_fields)

        print('Transforming numbers to float64')
        self.numeric_df = pd.DataFrame({column: to_numeric(self.df[column])
                                        for column in tqdm(self.df.columns)},
                                       index=self.df.index).astype(np.float64)

        self.numeric_df_copy = self.numeric_df.copy()

        self.load_column_to_id(list_of_columns_path, self.df.columns)

    def downsample_classes(self):
        super().downsample_classes()
        self.numeric_df = self.numeric_df_copy.iloc[self.indices].reset_index(drop=True)

    def upsample_classes(self):
        super().upsample_classes()
        self.numeric_df = self.numeric_df_copy.iloc[self.indices].reset_index(drop=True)

    def select_columns(self, column_names: List[str]):
        super().select_columns(column_names)

        numeric_column_names = list(set(column_names).intersection(set(self.numeric_df.columns)))
        self.numeric_df = self.numeric_df[numeric_column_names]

    def __getitem__(self, index):
        if index >= len(self.df):
            # Raise IndexError rather than the default KeyError, so to signal to the PyTorch DataLoader
            # that it should transform it to a StopIterationError.
            raise IndexError('Out of bounds!')

        raw_row = self.df.loc[index]
        is_tenant_id_present = TENANT_ID_COL in raw_row.index
        tenant_id = raw_row[TENANT_ID_COL] if is_tenant_id_present else None
        if self.drop_tenant_id_column and is_tenant_id_present:
            del raw_row[TENANT_ID_COL]
        return tokenized_row_encoder(raw_row=raw_row,
                                     number_row=self.numeric_df.loc[index],
                                     target=self.get_target(index),
                                     column_to_id=self.column_to_id,
                                     tokenizer=self.tokenizer,
                                     tenant_id=tenant_id)


class GenericTokenizedDatasetOneTokenPerCell(GenericTokenizedDataset):
    def __init__(
            self,
            sentence_embedding_model_name: str,
            data: pd.DataFrame,
            target_fields: List[UseCaseTarget],
            input_features: Optional[List[str]] = None,
            use_number_percentiles:
        bool = False,  # For now, due to how inefficient computation of percentiles is, default to False
            zmq_port: int = 5555,
            max_seq_length: int = 128,
            sql_filename: Union[str, None] = 'wikipedia'):

        super().__init__(data=data, input_features=input_features, target_fields=target_fields)

        self.row_tokenizer = ModularizedRowTokenizer(sentence_embedding_model_name,
                                                     zmq_port=zmq_port,
                                                     use_number_percentiles=use_number_percentiles,
                                                     sql_filename=sql_filename)

        self.zmq_port = zmq_port

        if self.df.shape[1] > max_seq_length:
            print(
                '\n\nWARNING: The number of columns in the dataset is greater than the max_seq_length.\n'
                'In the future, we will implement a "smart" choice of a subset of columns only, for now we only print this warning.\n'
                '(the training should still work just fine, only be slower & different to what it is used to in SSL pretraining)\n\n'
            )

    def __getitem__(self, index):
        if index >= len(self.df):
            # Raise IndexError rather than the default KeyError, so to signal to the PyTorch DataLoader
            # that it should transform it to a StopIterationError.
            raise IndexError('Out of bounds!')

        raw_row = self.df.loc[index]
        is_tenant_id_present = TENANT_ID_COL in raw_row.index
        tenant_id = raw_row[TENANT_ID_COL] if is_tenant_id_present else None
        if self.drop_tenant_id_column and is_tenant_id_present:
            del raw_row[TENANT_ID_COL]
        tokenized_row = self.row_tokenizer(raw_row)
        tokenized_row['labels'] = self.get_target(index)
        tokenized_row['tenant_id'] = tenant_id

        return {'input_ids': tokenized_row}


class GenericTokenizedDatasetOneTokenPerCellLikeSSL(GenericTokenizedDatasetOneTokenPerCell):
    id_mappings: Dict[str, Dict[str, int]] = {}

    @staticmethod
    def format_data(input_row: pd.Series,
                    target: pd.Series,
                    row_tokenizer: ModularizedRowTokenizer,
                    id_mappings: Dict[str, Dict[str, int]],
                    drop_tenant_id_column: bool = True,
                    override_is_text: Optional[np.ndarray] = None):
        """
        input_row: pd.Series, the row of the dataframe, without the targets
        target: pd.Series, the target row (has as many elements as target fields)
        """
        assert not set(input_row.index).intersection(target.index), 'Target column found in input features!'
        input_row = pd.concat([input_row, target])

        tenant_id = input_row[TENANT_ID_COL] if TENANT_ID_COL in input_row.index else None
        if drop_tenant_id_column and TENANT_ID_COL in input_row.index:
            del input_row[TENANT_ID_COL]

        tokenized_row = row_tokenizer(input_row, override_is_text)
        do_mask = torch.zeros(len(input_row), dtype=torch.bool)
        do_mask[-len(target):] = True  # Mask the target columns
        false_tensor = torch.zeros_like(do_mask)
        labels = {k: torch.clone(v) for k, v in tokenized_row.items()}
        input_ids = mask_one_row(tokenized_row, other_ids=tokenized_row, do_mask=do_mask, do_replace=false_tensor)

        if target.isnull().all():
            human_readable_labels = None
        else:
            # Transform classification targets, and leave regression unchanged
            human_readable_labels = {
                target_name: id_mappings[target_name].get(str(value), UNKNOWN_CLASS_LABEL_ID)
                if target_name in id_mappings else value
                for target_name, value in target.to_dict().items()
            }

        return {
            'column_names': input_row.index.tolist(),
            'human_readable_labels': human_readable_labels,
            'input_ids': input_ids,
            'labels': labels,
            'loss_mask': do_mask,
            'tenant_id': tenant_id,
        }

    def __getitem__(self, index):
        if index >= len(self.df):
            # Raise IndexError rather than the default KeyError, so to signal to the PyTorch DataLoader
            # that it should transform it to a StopIterationError.
            raise IndexError('Out of bounds!')

        target_row = self.get_target(index)
        if target_row is None:
            target_row = pd.Series([np.nan] * len(self.target_fields), index=[t.column for t in self.target_fields])
        else:
            for t in self.target_fields:
                if t.task == 'regression':
                    target_row[t.column] = float(target_row[t.column])
                else:
                    assert t.task == 'classification'
                    target_row[t.column] = str(target_row[t.column])
        # For classification tasks, we need to override the is_text flag for the target columns
        # (if some classes are text and others are numbers, it's important to treat them as text)
        if any(tf.task == 'classification' for tf in self.target_fields):
            override_is_text = np.zeros(len(target_row), dtype=bool)
            column_to_index = {c: i for i, c in enumerate(target_row.index)}
            for tf in self.target_fields:
                if tf.task == 'classification':
                    override_is_text[column_to_index[tf.column]] = True
        else:
            override_is_text = None
        return self.format_data(self.df.loc[index],
                                target_row,
                                self.row_tokenizer,
                                self.id_mappings,
                                override_is_text=override_is_text)

    def convert_classes_to_ids(self, id_mappings: Dict[str, int], target_column: str):
        """
        Convert class labels to IDs which the model can understand.
        :param id_mappings: dict containing the mapping between class values and the numeric IDs, per target i.e. {'class0': 0, 'class1': 1}
        """
        print(f'_NOT_ converting target column {target_column} classes to IDs - but storing them to convert on the fly')
        self.id_mappings[target_column] = id_mappings
