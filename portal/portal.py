import argparse
import os
import pickle
import random
import warnings
from argparse import Namespace
from hashlib import sha224
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
from time import time
from typing import Literal, Optional, Union
import subprocess

import fsspec
import numpy as np
import pandas as pd
import torch

from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics, model_selection
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, StandardScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange
from transformers import get_linear_schedule_with_warmup

from portal.constants import CACHE_PATH, ModelSize, ModelSizeAction, ModelType, ModelTypeAction
from portal.data.one_token_per_cell import ModularizedRowTokenizer
from portal.data.one_token_per_cell.collate import collate_fn as pretraining_collate_fn
from portal.data.one_token_per_cell.collate import pad_list_of_dict
from portal.data.tabular_datasets import GenericTokenizedDatasetOneTokenPerCellLikeSSL
from portal.model.one_token_modules import (
    MultiHeadedOneTokenPerCellModel,
    MultiHeadedOneTokenPerCellTripletModel,
    OneTokenPerCellLikeSSLModel,
)

from portal.scripts.start_embedding_server import embedding_server_starter
from portal.utils.target_properties import TargetProperties

warnings.filterwarnings('ignore', message='.*Our suggested max number of worker in.*')



def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str)
    parser.add_argument('--model_size', default=ModelSize.base, action=ModelSizeAction)
    parser.add_argument('--model_type', default=ModelType.ONE_TOKEN_PER_CELL, action=ModelTypeAction)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('-lr', '--learning_rate', type=float, default=2e-5)
    parser.add_argument('-c', '--checkpoint_path', default=None)
    parser.add_argument('--regression_as_classification', action='store_true')
    parser.add_argument('-rl', '--regression_loss', default='mixed', choices=['cross_entropy', 'l2', 'mixed'])
    parser.add_argument('-lbs', '--run_subset', default=None, nargs='+')
    parser.add_argument('--is_cosine_similarity', action='store_true')
    parser.add_argument('--is_remove_ssl_heads', action='store_true')
    parser.add_argument('-d',
                        '--dataset',
                        default='carte',
                        choices=['numeric', 'carte', '50k_subsample'],
                        help='50k_subsample and numeric are totally equivalent')
    parser.add_argument('-ts',
                        '--train_size',
                        type=int,
                        default=None,
                        help='Train size for the train-test split. If skipped, defaults to 80% of the dataset.')
    parser.add_argument(
        '-b',
        '--bagging',
        default=None,
        choices=[
            None, 'with_replacement', 'without_replacement', 'with_replacement_and_columns',
            'without_replacement_and_columns', 'columns_only'
        ],
        help='Bagging strategy to use. If None, no bagging is used. '
        'If with_replacement, bagging is used with replacement ("real" bagging), by sampling 10 times as many data points from the train dataset as there are in the train dataset, and using what is never picked (due to replacement) for validation - this is on average 37% of the data. '
        'If without_replacement, bagging is used without replacement, by sampling 10 times 80% for training and 20% for validation, like CARTE does. '
        'If including columns in bagging, also columns are sampled. Each of the column from the train set has a flat 20% chance of being dropped. '
        'If columns_only, only columns are sampled - the train / validation split is always the same (the default one as with no bagging).'
    )
    parser.add_argument('--keep_top_perc_bags', type=float, default=1.0)
    parser.add_argument(
        '--validation_random_state',
        type=int,
        default=42,
        help='Random state to do the train/validation split. Can be set for "manual" bagging. Temporary only.')
    parser.add_argument('-o',
                        '--output_folder',
                        default=None,
                        type=str,
                        help='If provided, saves labels and predictions for the test dataset to this folder.')
    parser.add_argument('--is_multi_gpu', action='store_true')
    parser.add_argument('--proc_per_gpu', type=int, default=1)
    parser.add_argument('--gpu_idx', type=int, default=0)
    parser.add_argument('--proc_idx', type=int, default=0)
    parser.add_argument(
        '-rtn',
        '--regression_target_normalization',
        default=None,
        type=str,
        choices=['standard', 'power', 'power_no_normalization'],
        help='If set, applies a normalization transformation (in the sense of Scikit-Learn) to the target variable.')

    return parser.parse_args(args)


def get_train_test_sizes(train_size):
    if train_size is not None:
        test_size = None
        suffix = f'_{train_size}'
    else:
        test_size = .2
        suffix = ''
    return train_size, test_size, suffix


def get_key_list(run_subset: Optional[list] = None,
                 dataset: Literal['carte', 'numeric', '50k_subsample', 'vime'] = 'carte'):
    if dataset == 'numeric':
        dataset = '50k_subsample'

    files_list_path = f'data/fine_tune_data_{dataset}/files.txt'
    with open(files_list_path) as fp:
        key_list = fp.read().strip().split('\n')

    if run_subset is not None:
        key_list = [k for k in key_list if any(x in k for x in run_subset)]

    return key_list


def guess_num_workers(proc_per_gpu):
    if torch.cuda.is_available():
        num_cores = os.cpu_count()
        if num_cores is not None:
            return int(num_cores * 2 / torch.cuda.device_count() / proc_per_gpu)
        return 8
    return 0


def start_embedding_server(zmq_port=5555):
    embedding_server_starter.start('sentence-transformers/all-MiniLM-L6-v2',
                                   zmq_port=zmq_port)


class MyDataset(Dataset):
    def __init__(self,
                 train_df,
                 target_column: str,
                 name: str,
                 is_classification,
                 id_mappings,
                 classification_to_regression_map=None,
                 target_column_reg_save=None,
                 is_validation=False,
                 zmq_port=5555,
                 processed_dataset_cache=None):
        self.row_tokenizer = ModularizedRowTokenizer('sentence-transformers/all-MiniLM-L6-v2',
                                                     zmq_port=zmq_port,
                                                     use_number_percentiles=False)
        self.zmq_port = zmq_port
        self.classification_to_regression_map = classification_to_regression_map

        self.name = name
        self.target = train_df[target_column]
        self.df = train_df.drop([target_column], axis=1)
        if target_column_reg_save:
            self.target_regression = train_df[target_column_reg_save]
            self.df = self.df.drop([target_column_reg_save], axis=1)
            self.df_save = self.df
            self.target_save = self.target
        self.target_column_reg_save = target_column_reg_save
        self.target_column = target_column
        self.is_validation = is_validation
        self.is_classification = is_classification
        self.processed_dataset_cache = processed_dataset_cache

        if is_classification:
            # for the classification_to_regression task we have percentile classes from 0 to 100
            if classification_to_regression_map is None:
                self.convert_target(id_mappings)
        else:
            self.target = self.target.astype(np.float32)

        self.id_mappings = id_mappings

    def under_sample_frequent_percentiles(self):
        if not self.is_validation and self.classification_to_regression_map is not None:
            counts = self.target_save.value_counts()
            mean_bin = int(4 * counts.mean())
            counts[counts > mean_bin] = mean_bin
            samples_per_class = counts.to_dict()

            rand_sampler = RandomUnderSampler(sampling_strategy=samples_per_class)
            X_sampled, y_sampled = rand_sampler.fit_resample(  # type: ignore
                self.df_save.values, self.target_save.values)
            self.df = pd.DataFrame(X_sampled, columns=self.df_save.columns)
            self.target = pd.Series(y_sampled)  # type: ignore

    def convert_target(self, id_mappings):
        self.target = self.target.astype(str).apply(id_mappings[self.target_column].get)

    def __len__(self):
        return len(self.df)

    def get_from_cache_or_tokenize(self, index):
        # We use the string representation of the row as a key for the cache
        # Remark that we need to convert to dict to avoid that, for low rows, pandas "helpfully"
        # truncates them...
        string_row = str(self.df.loc[index].to_dict())
        hashed_identifier = sha224(string_row.encode()).hexdigest()

        cache_file_path = self.processed_dataset_cache.joinpath(f'{hashed_identifier}.pickle')
        if cache_file_path.exists():
            try:
                with open(cache_file_path, 'rb') as fp:
                    tokenized_row = pickle.load(fp)
            except:
                # Sometimes loading fails; in that case, we just re-tokenize
                # This is probably the case if there are two identical rows, so identical hashes,
                # being created at the same time? So one process tries to read while the other
                # is still writing.
                print('Error in loading pickle, re-tokenizing')
            else:
                return tokenized_row

        tokenized_row = self.row_tokenizer(self.df.loc[index])

        statvfs = os.statvfs(self.processed_dataset_cache)
        if statvfs.f_frsize * statvfs.f_bfree >= 1e8:
            # At least 100 MB of free space, do cache
            with open(cache_file_path, 'wb') as fp:
                pickle.dump(tokenized_row, fp)
        return tokenized_row

    def __getitem__(self, index):
        if index >= len(self.df):
            raise IndexError('Out of bounds!')

        tokenized_row = self.get_from_cache_or_tokenize(index)

        tokenized_row['labels'] = {self.target_column: self.target.loc[index]}

        if self.target_column_reg_save and self.is_validation:
            tokenized_row['labels_regression'] = {self.target_column_reg_save: self.target_regression.loc[index]}

        return {'input_ids': tokenized_row}


class MyDatasetTriplet(MyDataset):
    def __getitem__(self, index):
        if index >= len(self.df):
            raise IndexError('Out of bounds!')

        tokenized_row = self.get_from_cache_or_tokenize(index)
        tokenized_row['labels'] = {self.target_column: self.target.loc[index]}

        if self.target_column_reg_save:
            tokenized_row['labels_regression'] = {self.target_column_reg_save: self.target_regression.loc[index]}

        if self.is_validation == False:
            index_positive, index_negative = self.pick_triplet_indices(index)
            if index_positive is None:
                print('index_positive is None, skipping')
                return self.__getitem__(index + 1)
            positive_ids = self.get_from_cache_or_tokenize(index_positive)
            negative_ids = self.get_from_cache_or_tokenize(index_negative)
            triplet_loss_mask = torch.tensor([True])
            triplet_classes = torch.tensor([self.target.loc[index], self.target.loc[index_negative]])

            return {
                'input_ids': tokenized_row,
                'positive_ids': positive_ids,
                'negative_ids': negative_ids,
                'triplet_loss_mask': triplet_loss_mask,
                'triplet_classes': triplet_classes,
            }
        return {'input_ids': tokenized_row}

    def pick_triplet_indices(self, index):
        '''
        Positive rows are the ones that have the same value in the selected column.
        '''
        mask_same_values_in_column = self.target == self.target.loc[index]
        mask_different_values_in_column = ~mask_same_values_in_column
        # set mask element to False at the place of the current row.
        # We want to pick a different similar row, not the same one
        mask_same_values_in_column.at[self.df.loc[index].name] = False

        index_positive, index_positive = None, None
        if mask_same_values_in_column.sum() > 0 and mask_different_values_in_column.sum() > 0:
            index_positive = random.choice(mask_same_values_in_column.index[mask_same_values_in_column])
            index_negative = random.choice(mask_different_values_in_column.index[mask_different_values_in_column])
        return index_positive, index_negative


class MyDatasetLikeSSL(MyDataset):
    def convert_target(self, id_mappings):
        # Just converting to string if classification
        if self.is_classification:
            self.target = self.target.astype(str)
        else:
            self.target = self.target.astype(float)

    def __getitem__(self, index):
        if index >= len(self.df):
            # Raise IndexError rather than the default KeyError, so to signal to the PyTorch DataLoader
            # that it should transform it to a StopIterationError.
            raise IndexError('Out of bounds!')

        if self.is_classification:
            # Target is always a single column here, so we need to add 1
            override_is_text = np.zeros(len(self.df.columns) + 1, dtype=bool)
            override_is_text[-1] = True
        else:
            override_is_text = None

        return GenericTokenizedDatasetOneTokenPerCellLikeSSL.format_data(self.df.loc[index],
                                                                         self.target.to_frame().loc[index],
                                                                         self.row_tokenizer,
                                                                         self.id_mappings,
                                                                         override_is_text=override_is_text)


def collate_fn(samples):
    result = {}
    samples_without_labels = [{
        k: v
        for k, v in sample['input_ids'].items() if k != 'labels' and k != 'labels_regression'
    } for sample in samples]

    result['input_ids'] = pad_list_of_dict(samples_without_labels)
    #only add in labels if not none
    if samples[0]['input_ids']['labels'] is not None:
        label_cols = samples[0]['input_ids'].get('labels').keys()
        # In truth, here label_cols has always length 1
        result['labels'] = {
            col: torch.tensor([sample['input_ids']['labels'][col] for sample in samples])
            for col in label_cols
        }

    if samples[0]['input_ids'].get('labels_regression') is not None:
        label_cols = samples[0]['input_ids'].get('labels_regression').keys()
        result['labels_regression'] = {
            col: torch.tensor([sample['input_ids']['labels_regression'][col] for sample in samples])
            for col in label_cols
        }

    if samples[0].get('triplet_loss_mask') is not None:
        result['positive_ids'] = pad_list_of_dict([sample['positive_ids'] for sample in samples])
        result['negative_ids'] = pad_list_of_dict([sample['negative_ids'] for sample in samples])
        result['triplet_loss_mask'] = pad_sequence([sample['triplet_loss_mask'] for sample in samples],
                                                   batch_first=True)
        result['triplet_classes'] = pad_sequence([sample['triplet_classes'] for sample in samples], batch_first=True)

    return result


def collate_labels(samples, key='labels'):
    label_cols = samples[0].get(key).keys()
    return {col: torch.tensor([sample[key][col] for sample in samples]) for col in label_cols}


def collate_fn_like_ssl(samples):
    result = pretraining_collate_fn(samples)
    result['column_names'] = samples[0]['column_names']
    if samples[0].get('human_readable_labels') is not None:
        result['human_readable_labels'] = collate_labels(samples, key='human_readable_labels')
    return result


def to_device(d, device):
    if isinstance(d, dict):
        return {k: to_device(v, device) for k, v in d.items() if k not in ['column_names', 'human_readable_labels']}
    return d.to(device)


def extract_loss(result, is_like_ssl, is_classification):
    if not is_like_ssl:
        return result[0]
    if is_classification:
        return result['loss']['text']
    return result['loss']['number']


def extract_predictions(result,
                        is_like_ssl,
                        target_column,
                        batch,
                        model,
                        classification_to_regression_map,
                        as_probabilities=False):
    assert not (is_like_ssl and classification_to_regression_map), 'Unknown combination'
    if as_probabilities:
        if is_like_ssl or (classification_to_regression_map is not None):
            raise NotImplementedError('predict_proba not implemented for these cases yet')
        return model.extract_predictions(result[1], as_probabilities=True)[target_column].detach().cpu().numpy()

    if not is_like_ssl:
        eval_preds = model.extract_predictions(result[1])[target_column].detach().cpu().numpy().flatten()
        if classification_to_regression_map is not None:
            # map percentiles to regression values
            eval_preds = [classification_to_regression_map[e] for e in eval_preds]
    else:
        # From here, like SSL
        logits = result['logits']
        logits['column_names'] = batch['column_names']
        eval_preds = model.extract_predictions(logits)[target_column].detach().cpu().numpy().flatten()

    return eval_preds


def map_regression_to_classification(df, target_column, num_bins=100):
    percentile_thresholds = np.percentile(df[target_column], np.linspace(0, 100, num_bins + 1))
    df['percentiles'] = np.interp(df[target_column], percentile_thresholds, np.linspace(0, 1, num_bins + 1))
    df['percentiles'] = np.minimum((df['percentiles'] * num_bins).astype(int), num_bins - 1)
    classification_to_regression_map = dict(zip(range(num_bins + 1), percentile_thresholds))

    bins = [0] * num_bins
    for p in df['percentiles']:
        bins[int(p)] += 1

    # find bins into which we should merge small bins of size <= 2
    idx_to_merge = {}
    for idx, p in enumerate(bins):
        bin_count_threshold = 2
        if p <= bin_count_threshold and p > 0:
            left, right = idx, idx

            while left > 0 or right < len(bins) - 1:
                if bins[left] > bin_count_threshold and bins[left] >= bins[right]:
                    idx_to_merge[idx] = left
                    break
                elif bins[right] > bin_count_threshold:
                    idx_to_merge[idx] = right
                    break

                if left > 0:
                    left -= 1
                if right < len(bins) - 1:
                    right += 1

    # move percentiles values of bin size 1 to bigger bins
    for i, row in df.iterrows():
        if row['percentiles'] in idx_to_merge:
            df.at[i, 'percentiles'] = idx_to_merge[row['percentiles']]

    return df, classification_to_regression_map


def select_random_split_for_bagging(train_df, bagging, stratify, random_state):
    np.random.seed(random_state)
    if bagging in ['without_replacement', 'without_replacement_and_columns']:
        train, val = model_selection.train_test_split(train_df,
                                                      random_state=random_state,
                                                      test_size=0.1,
                                                      stratify=stratify)
    elif bagging in ['with_replacement', 'with_replacement_and_columns']:
        train = train_df.sample(frac=2, replace=True, random_state=random_state)
        val = train_df[~train_df.index.isin(train.index)]
    else:
        assert bagging == 'columns_only', 'Unknown bagging strategy'
        # Always same split (same as without bagging)
        train, val = model_selection.train_test_split(train_df, random_state=42, test_size=0.1, stratify=stratify)

    if bagging in ['without_replacement_and_columns', 'with_replacement_and_columns', 'columns_only']:
        mask = np.random.choice([True, False], size=train.shape[1], p=[0.8, 0.2])
        mask[-1] = True  # always keep the target column
        train = train.loc[:, mask]

    return train, val


def check_encoder_architecture_matching(state_dict, model):
    checkpoint_encoder_layers = {}
    for l in state_dict:
        if 'encoder.layer' in l:
            checkpoint_encoder_layers[l] = state_dict[l].size()

    model_encoder_layers = {}
    for l, x in model.state_dict().items():
        if 'encoder.layer' in l:
            model_encoder_layers[l] = x.size()

    return checkpoint_encoder_layers == model_encoder_layers


class DatasetMetadata:
    def __init__(self, key: str, trainer: 'Trainer'):
        self.key = key
        self.is_classification = '/classification/' in key

        self.df = pd.read_parquet(f'./data/{key}')

        self.classification_to_regression_map = None
        self.target_column_reg_save = None

        self.target_column = self.df.columns[-1]
        self.df = self.df.dropna(subset=self.target_column)

        if not self.is_classification:
            if trainer.model_type == ModelType.ONE_TOKEN_PER_CELL_TRIPLET or trainer.regression_as_classification:
                # Regression as classification. We map regression values to percentiles before splitting into train, test, val sets
                self.target_column_reg_save = self.target_column
                self.df, self.classification_to_regression_map = map_regression_to_classification(
                    self.df, self.target_column)
                self.target_column = 'percentiles'
                self.is_classification = True

        if trainer.train_size is not None and trainer.train_size > len(self.df) * .9:
            train_size = int(len(self.df) * .9)
        else:
            train_size = trainer.train_size

        self.train_df, self.test_df = model_selection.train_test_split(self.df,
                                                                       train_size=train_size,
                                                                       test_size=trainer.test_size,
                                                                       random_state=42,
                                                                       stratify=self.get_stratify(self.df))

        if not self.is_classification:
            if trainer.regression_target_normalization is None:
                # Identity transformer
                self.scaler = FunctionTransformer()
            elif trainer.regression_target_normalization == 'standard':
                self.scaler = StandardScaler()
            elif trainer.regression_target_normalization == 'power':
                self.scaler = PowerTransformer()
            elif trainer.regression_target_normalization == 'power_no_normalization':
                self.scaler = PowerTransformer(standardize=False)
            else:
                raise ValueError('Unknown normalization method')

            self.train_df[self.target_column] = self.scaler.fit_transform(
                self.train_df[self.target_column].values[:, None])
            # We do not transform the test data, otherwise the metrics are not comparable.
            # We rather transform the predictions back to the original space.
            # self.test_df[self.target_column] = scaler.transform(self.test_df[self.target_column].values[:, None])

        self.stratify = self.get_stratify(self.train_df)

        self.output_checkpoint_folder = mkdtemp(suffix=f'_{trainer.gpu_idx}_{trainer.proc_idx}')

        if self.is_classification:
            unique_classes = self.train_df[self.target_column].unique()
            self.id_mappings = {self.target_column: {str(k): i for i, k in enumerate(unique_classes)}}
        else:
            self.id_mappings = {}

    def inverse_scaler(self, y):
        if self.is_classification:
            return y
        y = np.asarray(y).reshape(-1, 1)
        y = self.scaler.inverse_transform(y)
        return np.asarray(y).flatten()

    def get_stratify(self, df):
        if self.is_classification and self.classification_to_regression_map is None:
            return df[self.target_column]
        return None


class Trainer:
    def __init__(self,
                 run_name,
                 model_size,
                 model_type,
                 batch_size=32,
                 num_epochs=50,
                 patience=5,
                 warmup_epochs=2,
                 lr=2e-5,
                 regression_as_classification=False,
                 checkpoint_path=None,
                 regression_loss: Literal['l2', 'mixed', 'cross_entropy'] = 'l2',
                 is_cosine_similarity=False,
                 is_remove_ssl_heads=False,
                 train_size=None,
                 test_size: Optional[float] = .2,
                 bagging=None,
                 validation_random_state=42,
                 output_folder=None,
                 device=None,
                 processed_dataset_cache=None,
                 proc_per_gpu=1,
                 gpu_idx=0,
                 proc_idx=0,
                 keep_top_perc_bags=1.0,
                 weight_decay=0.01,
                 dropout_rate=0.1,
                 regression_target_normalization=None):
        self.run_name = run_name
        self.output_subfolder_name = run_name
        self.model_size = model_size
        self.model_type = model_type
        self.batch_size = batch_size

        if model_type == ModelType.ONE_TOKEN_PER_CELL_TRIPLET:
            # as we have anchor, positive, negative examples so during training we have 3 x batch_size
            self.batch_size = 16

        self.num_epochs = num_epochs
        self.patience = patience
        self.warmup_epochs = warmup_epochs
        self.lr = lr
        self.regression_as_classification = regression_as_classification
        self.checkpoint_path = checkpoint_path
        self.regression_loss: Literal['l2', 'mixed', 'cross_entropy'] = regression_loss
        self.is_cosine_similarity = is_cosine_similarity
        self.is_remove_ssl_heads = is_remove_ssl_heads
        self.train_size = train_size
        self.test_size = test_size
        self.bagging = bagging
        self.keep_top_perc_bags = keep_top_perc_bags
        self.validation_random_state = validation_random_state
        self.output_folder = output_folder
        self.processed_dataset_cache = processed_dataset_cache
        self.proc_per_gpu = proc_per_gpu
        self.gpu_idx = gpu_idx
        self.proc_idx = proc_idx
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.regression_target_normalization = regression_target_normalization

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
            print('self.device', self.device)

        # The following will be set separately for each training run:
        self.metadata: Union[DatasetMetadata, None] = None

    def train(self, key):
        if self.processed_dataset_cache.exists():
            rmtree(self.processed_dataset_cache)
        self.processed_dataset_cache.mkdir()
        self.metadata = DatasetMetadata(key, self)

        if self.bagging is None:
            # Without bagging, we just save the results as the name,
            # without any random state suffix.
            # That's because even if the random state is set to 42, the splitting procedure
            # is slightly different when using bagging, so we don't want to overwrite.
            self.output_subfolder_name = self.run_name
            train, val = model_selection.train_test_split(self.metadata.train_df,
                                                          random_state=self.validation_random_state,
                                                          test_size=0.1,
                                                          stratify=self.metadata.stratify)
            model, train_dataloader = self.train_model_on_one_split(train, val)
            test_loss, metric = self.predict_and_evaluate(model, self.metadata.test_df, 'test', train_dataloader)
        else:
            test_labels = None
            test_preds = []
            test_losses = []

            metric_per_bag = []
            number_of_bags = 10
            for random_state in trange(number_of_bags, leave=False):
                print('Bagging step:', random_state)
                self.output_subfolder_name = f'{self.run_name}_{random_state}_{self.bagging}'
                train, val = select_random_split_for_bagging(self.metadata.train_df, self.bagging,
                                                             self.metadata.stratify, random_state)
                model, train_dataloader = self.train_model_on_one_split(train, val)

                these_labels, these_preds, test_loss = self.compute_predictions(model,
                                                                                self.metadata.test_df,
                                                                                'test',
                                                                                train_dataloader,
                                                                                as_probabilities=True)

                these_labels = np.array(these_labels)
                if not self.metadata.is_classification:
                    these_preds = self.metadata.inverse_scaler(these_preds)

                if test_labels is None:
                    test_labels = these_labels
                else:
                    assert np.all(test_labels == these_labels), 'Labels do not match'
                test_preds.append(these_preds)
                test_losses.append(test_loss)

                if self.metadata.is_classification:
                    these_preds = np.argmax(these_preds, axis=1)
                metric = self.evaluate(test_labels, these_preds)
                metric_per_bag.append(metric)

            keep_n_bags = int(self.keep_top_perc_bags * number_of_bags)
            print('keep_n_bags', keep_n_bags)
            sorted_index = np.argsort(metric_per_bag)[::-1][:keep_n_bags]
            test_losses = np.asarray(test_losses)[sorted_index]
            test_preds = np.asarray(test_preds)[sorted_index]

            test_loss = np.mean(test_losses)
            # test_preds is a list of 10 elements, each being:
            # - for regression, just a list of scalar
            # - for classification, a list of arrays of probabilities
            # In the first case, we just need to average. In the second case, we average the probabilities and then take the argmax
            test_preds = np.mean(test_preds, axis=0)
            if self.metadata.is_classification:
                test_preds = np.argmax(test_preds, axis=1)
            metric = self.evaluate(test_labels, test_preds)

        rmtree(self.metadata.output_checkpoint_folder)
        rmtree(self.processed_dataset_cache)
        return test_loss, metric

    def build_model(self, target_to_properties):
        assert isinstance(self.metadata, DatasetMetadata)
        if self.model_type == ModelType.ONE_TOKEN_PER_CELL:
            model = MultiHeadedOneTokenPerCellModel(target_to_properties,
                                                    model_size=self.model_size,
                                                    dropout_rate=self.dropout_rate)
            is_like_ssl = False
        elif self.model_type == ModelType.ONE_TOKEN_PER_CELL_TRIPLET:
            model = MultiHeadedOneTokenPerCellTripletModel(
                target_to_properties,
                model_size=self.model_size,
                regression_as_classification=self.metadata.classification_to_regression_map is not None,
                is_cosine_similarity=self.is_cosine_similarity,
                dropout_rate=self.dropout_rate)
            is_like_ssl = False
        elif self.model_type == ModelType.ONE_TOKEN_PER_CELL_LIKE_SSL:
            model = OneTokenPerCellLikeSSLModel(target_to_properties,
                                                model_size=self.model_size,
                                                regression_type=self.regression_loss,
                                                dropout_rate=self.dropout_rate)
            is_like_ssl = True
            if self.checkpoint_path is None:
                print('\n\nWARNING: OneTokenPerCellLikeSSL should really use a pretrained checkpoint.\n\n')
        else:
            raise ValueError(f'Unsupported model type: {self.model_type}')

        model = model.to(self.device)

        if self.checkpoint_path is not None:
            with fsspec.open(self.checkpoint_path, 'rb') as f:
                state_dict = torch.load(f, map_location=self.device)  # type: ignore
                if self.is_remove_ssl_heads:
                    for l in list(state_dict.keys()):
                        if 'number_head' in l or 'date_head' in l or 'text_head' in l:
                            state_dict.pop(l)

            assert check_encoder_architecture_matching(state_dict, model), 'Model encoder architecture different than the one in the checkpoint'
            model.load_state_dict(state_dict, strict=False)

        return model, is_like_ssl

    def train_model_on_one_split(self, train, val):
        assert isinstance(self.metadata, DatasetMetadata)

        max_columns = 300
        if len(train.columns) > max_columns:
            # if there are more than `max_columns` columns then randomly sample `max_columns` columns, otherwise GPU A10 (with 24GiB) goes out of memory
            df_columns = list(train.columns[:-1].values)
            random.seed(10)
            sampled_columns = random.sample(df_columns, max_columns)
            train = train[sampled_columns + [self.metadata.target_column]]

        train_dataset = self.get_dataset(train, 'train')
        train_dataloader = self.get_dataloader(train_dataset, shuffle=True)

        if self.metadata.is_classification:
            if self.metadata.classification_to_regression_map is not None:
                target_to_properties = {
                    self.metadata.target_column:
                    TargetProperties('classification', classes=len(self.metadata.classification_to_regression_map))
                }
            else:
                target_to_properties = {
                    self.metadata.target_column:
                    TargetProperties('classification',
                                     classes=list(self.metadata.id_mappings[self.metadata.target_column]),
                                     string_tokenizer=train_dataset.row_tokenizer.string_tokenizer)
                }
        else:
            target_to_properties = {
                self.metadata.target_column: TargetProperties('regression', regression_type=self.regression_loss)
            }

        model, is_like_ssl = self.build_model(target_to_properties)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(self.warmup_epochs * len(train_dataloader)),
            num_training_steps=(len(train_dataloader) * self.num_epochs),
        )

        valid_metrics = []
        valid_losses = []
        patience_count = 0
        best_ckpt_path = os.path.join(self.metadata.output_checkpoint_folder, 'best_ckpt.pt')
        print('Saving to best_ckpt_path:', best_ckpt_path)

        with trange(self.num_epochs) as progress_bar:
            for epoch_idx in progress_bar:
                model = model.train()
                if self.metadata.classification_to_regression_map:
                    train_dataloader.dataset.under_sample_frequent_percentiles()  # type: ignore

                for batch in tqdm(train_dataloader, leave=False):
                    result = model(**to_device(batch, self.device))
                    loss = extract_loss(result, is_like_ssl, self.metadata.is_classification)
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if self.model_type == ModelType.ONE_TOKEN_PER_CELL_TRIPLET:
                    if epoch_idx % 3 != 0 or epoch_idx == 0:
                        # for triplet loss, evaluate only every 3rd epoch as evaluation is expensive
                        # because we generate reference embeddings for the entire training set
                        continue

                loss, metric = self.predict_and_evaluate(model, val, 'val', train_dataloader)
                valid_metrics.append(metric)
                valid_losses.append(loss)
                if len(valid_metrics) == 1 or metric > max(valid_metrics[:-1]):
                    patience_count = 0
                    torch.save(model.state_dict(), best_ckpt_path)
                else:
                    patience_count += 1
                if patience_count == self.patience:
                    print('Reached patience limit, stopping')
                    break

                last_metrics = ' '.join([f'{x:.2%}' for x in valid_metrics[-3:]])
                progress_bar.set_description(f'Last 3 valid metrics: {last_metrics}, best: {max(valid_metrics):.2%}')

        checkpoint = torch.load(best_ckpt_path)
        model.load_state_dict(checkpoint)

        return model, train_dataloader

    def compute_predictions(self, model, df, name: str, train_dataloader, as_probabilities):
        assert isinstance(self.metadata, DatasetMetadata)
        dataloader = self.get_dataloader(self.get_dataset(df, name, is_validation=True), shuffle=False)

        model.eval()
        eval_labels = []
        eval_preds = []
        eval_losses = []

        if self.metadata.target_column_reg_save:
            assert self.metadata.classification_to_regression_map is not None, 'classification_to_regression_map should not be None for regression like classification'
        else:
            classification_to_regression_map = None  # is this superfluous?

        if isinstance(model, MultiHeadedOneTokenPerCellTripletModel):
            if as_probabilities:
                raise NotImplementedError('Not implemented yet')
            number_of_classes = model.target_to_properties[self.metadata.target_column].size
            pred_per_class = torch.zeros((number_of_classes, model.model_size[1]))
            counter_per_class = torch.zeros((number_of_classes))

            # generate centers of classes for the training set, a little bit too time consuming
            for batch in tqdm(train_dataloader, leave=False, desc='Reference class centers generation'):
                with torch.no_grad():
                    batch = to_device(batch, self.device)
                    batch.update({'is_validation': True})
                    pred = model(**batch)[0]
                    for l, p in zip(batch['labels'][self.metadata.target_column].detach().cpu(),
                                    pred['triplet'].detach().float().cpu()):
                        pred_per_class[l.item()] += p
                        counter_per_class[l.item()] += 1

            counter_per_class[counter_per_class == 0] = 1
            pred_per_class = pred_per_class / counter_per_class.unsqueeze(1)
            cosine_similarity = torch.nn.CosineSimilarity(dim=2)

            for batch in tqdm(dataloader, leave=False):
                with torch.no_grad():
                    batch_updated = to_device(batch, self.device)
                    batch_updated.update({'is_validation': True})
                    pred = model(**batch_updated)[0]

                if model.is_cosine_similarity:
                    similarities = cosine_similarity(pred['triplet'].detach().float().cpu().unsqueeze(1),
                                                     pred_per_class.unsqueeze(0))
                    sim = torch.argmax(similarities, dim=1)
                else:
                    pred_examples = pred['triplet'].detach().float().cpu().unsqueeze(1)
                    reference_classes_preds = pred_per_class.unsqueeze(0)
                    similarities = (pred_examples - reference_classes_preds).pow(2).sum(2).sqrt()
                    sim = torch.argmin(similarities, dim=1)

                eval_preds.extend(sim.numpy().flatten())
                if self.metadata.target_column_reg_save:
                    # use real regression values as labels
                    eval_labels.extend(batch['labels_regression'][self.metadata.target_column_reg_save].numpy())
                else:
                    eval_labels.extend(batch['labels'][self.metadata.target_column].numpy())
            loss = None
        else:
            is_like_ssl = isinstance(model, OneTokenPerCellLikeSSLModel)
            for batch in tqdm(dataloader, leave=False):
                with torch.no_grad():
                    result = model(**to_device(batch, self.device))
                loss = extract_loss(result, is_like_ssl, self.metadata.is_classification)
                preds = extract_predictions(result,
                                            is_like_ssl,
                                            self.metadata.target_column,
                                            batch,
                                            model,
                                            classification_to_regression_map=classification_to_regression_map,
                                            as_probabilities=as_probabilities)
                eval_preds.extend(preds)

                labels_target_column = self.metadata.target_column
                if self.metadata.target_column_reg_save:
                    # use real regression values as labels (not percentiles)
                    labels_key = 'labels_regression'
                    labels_target_column = self.metadata.target_column_reg_save
                elif is_like_ssl:
                    labels_key = 'human_readable_labels'
                else:
                    labels_key = 'labels'

                eval_labels.extend(batch[labels_key][labels_target_column].numpy())
                eval_losses.append(loss.item())
            loss = np.mean(eval_losses)

        if self.output_folder is not None:
            target_folder = Path(self.output_folder).joinpath(self.output_subfolder_name).joinpath(
                self.metadata.key).with_suffix('')
            target_folder.mkdir(exist_ok=True, parents=True)
            with open(target_folder.joinpath(f'{name}_labels.pickle'), 'wb') as fp:
                pickle.dump(eval_labels, fp)
            with open(target_folder.joinpath(f'{name}_preds.pickle'), 'wb') as fp:
                pickle.dump(eval_preds, fp)

        return eval_labels, eval_preds, loss

    def predict_and_evaluate(self, model, dataset_to_be_predicted, name: str, train_dataloader):
        eval_labels, eval_preds, test_loss = self.compute_predictions(model,
                                                                      dataset_to_be_predicted,
                                                                      name,
                                                                      train_dataloader,
                                                                      as_probabilities=False)

        assert isinstance(self.metadata, DatasetMetadata)

        if not self.metadata.is_classification:
            eval_preds = self.metadata.inverse_scaler(eval_preds)
            if name == 'val':
                # For validation, we also need to transfom the labels back; not for test
                eval_labels = self.metadata.inverse_scaler(eval_labels)

        return test_loss, self.evaluate(eval_labels, eval_preds)

    def evaluate(self, eval_labels, eval_preds):
        assert isinstance(self.metadata, DatasetMetadata)
        if self.metadata.is_classification and self.metadata.target_column_reg_save is None:
            metric = metrics.accuracy_score(eval_labels, eval_preds)
        else:
            metric = metrics.r2_score(eval_labels, eval_preds)
        return metric

    def get_dataset(self, df, name, is_validation=False):
        assert isinstance(self.metadata, DatasetMetadata)
        if self.model_type == ModelType.ONE_TOKEN_PER_CELL_LIKE_SSL:
            class_name = MyDatasetLikeSSL
        elif self.model_type == ModelType.ONE_TOKEN_PER_CELL_TRIPLET:
            class_name = MyDatasetTriplet
        else:
            class_name = MyDataset

        return class_name(df.reset_index(drop=True),
                          self.metadata.target_column,
                          is_classification=self.metadata.is_classification,
                          id_mappings=self.metadata.id_mappings,
                          classification_to_regression_map=self.metadata.classification_to_regression_map,
                          target_column_reg_save=self.metadata.target_column_reg_save,
                          is_validation=is_validation,
                          name=name,
                          processed_dataset_cache=self.processed_dataset_cache)

    def get_dataloader(self, dataset, shuffle):
        if self.model_type == ModelType.ONE_TOKEN_PER_CELL_LIKE_SSL:
            this_collate_fn = collate_fn_like_ssl
        else:
            this_collate_fn = collate_fn
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=shuffle,
                          collate_fn=this_collate_fn,
                          num_workers=guess_num_workers(self.proc_per_gpu))


def main(run_name,
         model_size,
         model_type,
         max_epochs,
         patience,
         batch_size,
         warmup_epochs,
         learning_rate,
         regression_as_classification=False,
         checkpoint_path=None,
         regression_loss: Literal['l2', 'mixed', 'cross_entropy'] = 'l2',
         run_subset=None,
         is_cosine_similarity=False,
         is_remove_ssl_heads=False,
         dataset: Literal['carte', '50k_subsample', 'numeric', 'vime'] = 'carte',
         train_size=None,
         bagging=None,
         validation_random_state=42,
         output_folder=None,
         gpu_idx=0,
         proc_per_gpu=1,
         proc_idx=0,
         keep_top_perc_bags=1.0,
         weight_decay=0.1,
         dropout_rate=0.1,
         regression_target_normalization=None):
    device = torch.device(f'cuda:{gpu_idx}')
    print('device', device)

    train_size, test_size, suffix = get_train_test_sizes(train_size)

    # Add a random suffix, should we ever run two jobs on the same machine.
    processed_dataset_cache = CACHE_PATH.joinpath(f'processed_dataset_cache_{gpu_idx}_{proc_idx}_{random.randint(0, int(1e9))}')

    key_list = get_key_list(run_subset=run_subset,
                            dataset=dataset)
    start_embedding_server()
    progress_bar = tqdm(key_list, desc='Looping through datasets')

    trainer = Trainer(run_name,
                      model_size,
                      model_type,
                      batch_size=batch_size,
                      num_epochs=max_epochs,
                      patience=patience,
                      warmup_epochs=warmup_epochs,
                      lr=learning_rate,
                      regression_as_classification=regression_as_classification,
                      checkpoint_path=checkpoint_path,
                      regression_loss=regression_loss,
                      is_cosine_similarity=is_cosine_similarity,
                      is_remove_ssl_heads=is_remove_ssl_heads,
                      train_size=train_size,
                      test_size=test_size,
                      bagging=bagging,
                      validation_random_state=validation_random_state,
                      output_folder=output_folder,
                      processed_dataset_cache=processed_dataset_cache,
                      device=device,
                      proc_per_gpu=proc_per_gpu,
                      gpu_idx=gpu_idx,
                      proc_idx=proc_idx,
                      keep_top_perc_bags=keep_top_perc_bags,
                      weight_decay=weight_decay,
                      dropout_rate=dropout_rate,
                      regression_target_normalization=regression_target_normalization)

    all_metrics = []

    for key in progress_bar:
        progress_bar.set_postfix({'key': key.split('/')[-1].replace('.parquet', '')})
        time_start = time()
        try:
            _, test_metric = trainer.train(key)
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print(f'Skipping {key} due to error: {e}')
            import traceback
            print(traceback.format_exc())
            continue
        metric_name = 'accuracy' if '/classification/' in key else 'r2'
        dataset_name = key.split('/')[-1][:-8]
        total_runtime = time() - time_start

        print(f'{dataset_name}{suffix}_{metric_name}', test_metric)
        print(f'{dataset_name}{suffix}_runtime', total_runtime)
        with open(f"{run_name}_results.txt", "a") as results_file:
            results_file.write(f'{dataset_name}{suffix}_{metric_name}: {test_metric}\n')
        all_metrics.append(test_metric)

    return sum(all_metrics) / len(all_metrics)


def main_multi_gpu(args):
    _, _, suffix = get_train_test_sizes(args.train_size)

    key_list = get_key_list(run_subset=args.run_subset,
                            dataset=args.dataset)
    start_embedding_server()
    print(key_list)

    key_list_buckets = []
    number_of_gpus = torch.cuda.device_count()
    print('Number of gpus:', number_of_gpus)
    proc_per_gpu = args.proc_per_gpu
    number_of_proc = number_of_gpus * proc_per_gpu

    # use at least one dataset per process
    datasets_per_proc = max(int(len(key_list) / float(number_of_proc)), 1)
    # shuffle key_list to get more even runtime distribution per process
    # some smarter shuffling can be used not to put large datasets in the same process/gpu
    random.shuffle(key_list)
    for i in range(number_of_proc):
        key_list_buckets.append(key_list[i * datasets_per_proc : (i + 1) * datasets_per_proc])

    # add remaining datasets (1 extra dataset per process until we distribute all datasets)
    j = 0
    for i in range(number_of_proc * datasets_per_proc, len(key_list)):
        key_list_buckets[j].append(key_list[i])
        j += 1

    # set is_multi_gpu to False so that from now we run a single experiment for subset of datasets
    args.is_multi_gpu = False

    arg_str = ''
    args_dict = vars(args)
    args_dict.pop('run_name')
    args_dict.pop('proc_per_gpu')
    args_dict.pop('run_subset')
    # map args to string, probably can be done in simpler way
    for k, v in args_dict.items():
        if v is None or v == False:
            continue
        if k == 'model_size':
            v = v.name
        elif k == 'model_type':
            v = v.value

        if isinstance(v, bool) and v == True:
            arg_str += '--' + k + ' '
        else:
            arg_str += '--' + k + '=' + str(v) + ' '

    commands = []
    base_command = f'python3 -m portal.portal {args.run_name} '

    print('key_list_buckets', key_list_buckets)
    for proc_idx, key_list in enumerate(key_list_buckets):
        if len(key_list) == 0:
            continue
        run_subset = ''
        print('Number of datasets per one finetuning process:', len(key_list))
        for k in key_list:
            run_subset += k + ' '
        commands.append(base_command + arg_str.strip() + ' ' + \
                        '--run_subset ' + run_subset.strip() + ' ' + \
                        '--gpu_idx=' + str(proc_idx // proc_per_gpu) + ' ' + \
                        '--proc_per_gpu=' + str(proc_per_gpu) + ' ' + \
                        '--proc_idx=' + str(proc_idx))

    print('Commands:\n', *commands, sep='\n')
    procs = [subprocess.Popen(i, shell=True) for i in commands]
    for p in procs:
        p.wait()


if __name__ == '__main__':
    args = parse_args()
    if args.is_multi_gpu:
        main_multi_gpu(args)
    else:
        main(args.run_name,
             args.model_size,
             args.model_type,
             args.max_epochs,
             args.patience,
             args.batch_size,
             args.warmup_epochs,
             args.learning_rate,
             regression_as_classification=args.regression_as_classification,
             checkpoint_path=args.checkpoint_path,
             regression_loss=args.regression_loss,
             run_subset=args.run_subset,
             is_cosine_similarity=args.is_cosine_similarity,
             is_remove_ssl_heads=args.is_remove_ssl_heads,
             dataset=args.dataset,
             train_size=args.train_size,
             bagging=args.bagging,
             validation_random_state=args.validation_random_state,
             output_folder=args.output_folder,
             gpu_idx=args.gpu_idx,
             proc_per_gpu=args.proc_per_gpu,
             proc_idx=args.proc_idx,
             keep_top_perc_bags=args.keep_top_perc_bags,
             regression_target_normalization=args.regression_target_normalization)
