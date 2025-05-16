import random
from abc import ABC
from pathlib import Path
from shutil import rmtree
from typing import Union

import numpy as np
import pandas as pd
import torch
from sklearn import metrics, model_selection
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import get_linear_schedule_with_warmup

from portal.constants import CACHE_PATH, ModelSize
from portal.portal import (
    MultiHeadedOneTokenPerCellModel,
    MyDataset,
    TargetProperties,
    check_encoder_architecture_matching,
    collate_fn,
    extract_predictions,
    guess_num_workers,
    start_embedding_server,
    to_device,
)


def get_dataset(df, name, is_validation, target_column, is_classification,
                id_mappings, processed_dataset_cache):
    return MyDataset(df.reset_index(drop=True),
                     target_column,
                     is_classification=is_classification,
                     id_mappings=id_mappings,
                     classification_to_regression_map=None,
                     target_column_reg_save=None,
                     is_validation=is_validation,
                     name=name,
                     processed_dataset_cache=processed_dataset_cache)


def get_dataloader(dataset, shuffle):
    return DataLoader(dataset,
                      batch_size=32,
                      shuffle=shuffle,
                      collate_fn=collate_fn,
                      num_workers=guess_num_workers(proc_per_gpu=1))


def compute_predictions(model, df, name: str, as_probabilities, target_column,
                        device, is_classification, id_mappings,
                        processed_dataset_cache):
    dataset = get_dataset(df,
                          name,
                          is_validation=True,
                          target_column=target_column,
                          is_classification=is_classification,
                          id_mappings=id_mappings,
                          processed_dataset_cache=processed_dataset_cache)
    dataloader = get_dataloader(dataset, shuffle=False)

    model.eval()
    eval_labels = []
    eval_preds = []
    eval_losses = []

    for batch in tqdm(dataloader, leave=False):
        with torch.no_grad():
            result = model(**to_device(batch, device))
        loss = result[0]
        preds = extract_predictions(result,
                                    target_column,
                                    model,
                                    classification_to_regression_map=None,
                                    as_probabilities=as_probabilities)
        eval_preds.extend(preds)

        labels_target_column = target_column
        labels_key = 'labels'

        eval_labels.extend(batch[labels_key][labels_target_column].numpy())
        eval_losses.append(loss.item())
    loss = np.mean(eval_losses)

    return eval_labels, eval_preds, loss


def predict_and_evaluate(model, dataset_to_be_predicted, name: str,
                         is_classification, scaler, target_column, device,
                         id_mappings, processed_dataset_cache):
    eval_labels, eval_preds, test_loss = compute_predictions(
        model,
        dataset_to_be_predicted,
        name,
        as_probabilities=False,
        target_column=target_column,
        device=device,
        is_classification=is_classification,
        id_mappings=id_mappings,
        processed_dataset_cache=processed_dataset_cache)

    if not is_classification:
        eval_preds = scaler.inverse_transform(
            np.array(eval_preds).reshape(-1, 1)).flatten()
        # if name == 'val':
        #     # For validation, we also need to transfom the labels back; not for test
        #     eval_labels = scaler.inverse_transform(np.array(eval_labels).reshape(-1, 1)).flatten()

    if is_classification:
        metric = metrics.accuracy_score(eval_labels, eval_preds)
    else:
        metric = metrics.r2_score(eval_labels, eval_preds)

    return test_loss, metric


def do_train(checkpoint_path: Union[Path, str, None],
             is_classification: bool,
             train: pd.DataFrame,
             val: pd.DataFrame,
             task_name: str,
             num_epochs=100,
             max_steps_per_epoch=None):
    start_embedding_server()
    processed_dataset_cache = CACHE_PATH.joinpath(
        f'processed_dataset_cache_{task_name}_{random.randint(0, int(1e9))}')

    if processed_dataset_cache.exists():
        rmtree(processed_dataset_cache)
    processed_dataset_cache.mkdir()

    target_column = train.columns[-1]

    max_columns = 300
    if len(train.columns) > max_columns:
        df_columns = list(train.columns[:-1].values)
        random.seed(10)
        sampled_columns = random.sample(df_columns, max_columns)
        train = train[sampled_columns + [target_column]]

    if is_classification:
        unique_classes = train[target_column].unique()
        scaler = FunctionTransformer()
        id_mappings = {
            target_column: {
                str(k): i
                for i, k in enumerate(unique_classes)
            }
        }
    else:
        id_mappings = {}
        scaler = StandardScaler()
        train[target_column] = scaler.fit_transform(
            train[target_column].values[:, None])

    train_dataset = get_dataset(train, 'train', False, target_column,
                                is_classification, id_mappings,
                                processed_dataset_cache)

    train_dataloader = get_dataloader(train_dataset, True)

    if is_classification:
        target_to_properties = {
            target_column:
            TargetProperties(
                'classification',
                classes=list(id_mappings[target_column]),
                string_tokenizer=train_dataset.row_tokenizer.string_tokenizer)
        }
    else:
        target_to_properties = {
            target_column: TargetProperties('regression', regression_type='l2')
        }

    device = torch.device('cuda')

    model = MultiHeadedOneTokenPerCellModel(target_to_properties,
                                            model_size=ModelSize.base,
                                            dropout_rate=0.1).to(device)

    if checkpoint_path is not None:
        # If None, we are training from scratch
        state_dict = torch.load(checkpoint_path, map_location=device)
    assert check_encoder_architecture_matching(
        state_dict, model
    ), 'Model encoder architecture different than the one in the checkpoint'
    model.load_state_dict(state_dict, strict=False)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=2e-5,
                                  weight_decay=0.01)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(2 * len(train_dataloader)),
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    valid_metrics = []
    valid_losses = []
    patience_count = 0
    best_state_dict = {}
    cpu = torch.device('cpu')
    with trange(num_epochs) as progress_bar:
        for _ in progress_bar:
            model.train()
            for i, batch in enumerate(tqdm(train_dataloader, leave=False)):
                result = model(**to_device(batch, device))
                loss = result[0]
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                if max_steps_per_epoch is not None and i >= max_steps_per_epoch:
                    break

            loss, metric = predict_and_evaluate(model, val, 'val',
                                                is_classification, scaler,
                                                target_column, device,
                                                id_mappings,
                                                processed_dataset_cache)
            valid_metrics.append(metric)
            valid_losses.append(loss)
            if len(valid_metrics) == 1 or metric > max(valid_metrics[:-1]):
                patience_count = 0
                best_state_dict = {
                    k: v.clone().to(cpu)
                    for k, v in model.state_dict().items()
                }
            else:
                patience_count += 1
            if patience_count == 10:
                print('Reached patience limit, stopping')
                break

            last_metrics = ' '.join([f'{x:.2%}' for x in valid_metrics[-3:]])
            progress_bar.set_description(
                f'Last 3 valid metrics: {last_metrics}, best: {max(valid_metrics):.2%}'
            )

    rmtree(processed_dataset_cache)

    return best_state_dict, max(valid_metrics), id_mappings, scaler


class PortalEstimator(BaseEstimator, ABC):
    is_classification: bool

    def __init__(self,
                 checkpoint_path: Union[Path, str, None],
                 model_size=ModelSize.base,
                 bagging=6,
                 keep_top_perc_bags=0.5,
                 num_epochs=100,
                 max_steps_per_epoch=1000):
        self.checkpoint_path = checkpoint_path
        self.model_size = model_size
        self.bagging = bagging
        self.keep_top_perc_bags = keep_top_perc_bags
        self.num_epochs = num_epochs
        self.max_steps_per_epoch = max_steps_per_epoch
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y, name='target')

        self.feature_columns = X.columns
        self.target_column = y.name
        df_train = pd.concat([X, y.to_frame()], axis=1)

        self.best_state_dict = []
        self.best_metric = []
        self.id_mappings = []
        self.scaler = []

        for i in trange(self.bagging):
            if self.bagging > 1:
                train = df_train.sample(frac=2, replace=True, random_state=i)
                val = df_train[~df_train.index.isin(train.index)]
                # limit size of val to at most 3200 rows, i.e. 100 batches, to limit validation time
                if len(val) > 3200:
                    val = val.sample(3200, random_state=i, replace=False)
            else:
                test_size = min(int(0.1 * len(df_train)), 3200)
                train, val = model_selection.train_test_split(
                    df_train, random_state=42, test_size=test_size)

            bsd, bm, im, s = do_train(
                self.checkpoint_path,
                self.is_classification,
                train,
                val,
                self.name,
                self.num_epochs,
                max_steps_per_epoch=self.max_steps_per_epoch)
            self.best_state_dict.append(bsd)
            self.best_metric.append(bm)
            self.id_mappings.append(im)
            self.scaler.append(s)

        if self.keep_top_perc_bags < 1.0 and self.bagging > 1:
            top_k = max(1, int(round(self.bagging * self.keep_top_perc_bags)))
            best_indices = np.argsort(self.best_metric)[-top_k:]
            self.best_state_dict = [
                self.best_state_dict[i] for i in best_indices
            ]
            self.best_metric = [self.best_metric[i] for i in best_indices]
            self.id_mappings = [self.id_mappings[i] for i in best_indices]
            self.scaler = [self.scaler[i] for i in best_indices]

        return self

    def _predict(self, X):
        preds = 0
        for i in range(len(self.best_state_dict)):
            preds += self._predict_one_bag(X, self.best_state_dict[i],
                                            self.id_mappings[i],
                                            self.scaler[i])

        return preds / len(self.best_state_dict)


class PortalClassifier(BaseEstimator, ClassifierMixin):
    is_classification = True

    def fit(self, X, y):
        self._classes = sorted(set(y.values), key=lambda x: str(x))
        return super().fit(X, y)

    def _predict_one_bag(self, X, state_dict, id_mappings, scaler):
        dataset = get_dataset(X,
                              'test',
                              is_validation=True,
                              target_column=self.target_column,
                              is_classification=self.is_classification,
                              id_mappings=id_mappings,
                              processed_dataset_cache=CACHE_PATH)
        target_to_properties = {
            self.target_column:
            TargetProperties(
                'classification',
                classes=list(id_mappings[self.target_column]),
                string_tokenizer=dataset.row_tokenizer.string_tokenizer)
        }

        model = MultiHeadedOneTokenPerCellModel(
            target_to_properties=target_to_properties,
            model_size=self.model_size,
            dropout_rate=0.1).to(self.device)
        model.load_state_dict(state_dict, strict=True)

        _, eval_preds, _ = compute_predictions(
            model,
            X,
            'test',
            as_probabilities=True,
            target_column=self.target_column,
            device=self.device,
            is_classification=self.is_classification,
            id_mappings=id_mappings,
            processed_dataset_cache=CACHE_PATH)

        # Prediction are probabilities of shape (num_samples, num_classes)
        # They are ordered with the order given by id_mappings, which might be e.g.
        # {'Rating': {'1.0': 0, '0.0': 1}}
        # (if here Rating is the target column)
        # So we need to sort the classes in the order given by self._classes
        eval_preds = np.asarray(eval_preds)
        class_to_idx = id_mappings[self.target_column]
        permutation = []
        for class_name in self._classes:
            if class_name in class_to_idx:
                permutation.append(class_to_idx[class_name])
            elif str(class_name) in class_to_idx:
                permutation.append(class_to_idx[str(class_name)])
            else:
                permutation.append(-1)
        num_missing = sum(x == -1 for x in permutation)
        if num_missing + len(class_to_idx) != len(self._classes):
            raise ValueError(
                f'Class names {self._classes} not matching class_to_idx {class_to_idx}, check types! {[type(class_name) for class_name in self._classes]} vs {[type(x) for x in class_to_idx]}'
            )

        eval_preds = eval_preds[:, permutation]
        # Set 0 prob to missing classes
        eval_preds[:, np.array(permutation) == -1] = 0
        return np.asarray(eval_preds)

    def predict_proba(self, X):
        check_is_fitted(self)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_columns)

        first_class = next(iter(self.id_mappings[0][self.target_column]))
        masked_df = pd.concat(
            [X, pd.DataFrame({self.target_column: [first_class for _ in range(len(X))]}, index=X.index)], axis=1)
        
        return self._predict(masked_df, True)

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_columns)

        prediction_probabilities = self.predict_proba(X)
        preds_indices = prediction_probabilities.argmax(axis=-1)
        id_to_class = {i: v for i, v in enumerate(self._classes)}
        preds = [id_to_class[p] for p in preds_indices]

        return pd.Series(preds, name=self.target_column, index=X.index)


class PortalRegressor(BaseEstimator, RegressorMixin):

    def _predict_one_bag(self, X, state_dict, id_mappings, scaler):
        target_to_properties = {
            self.target_column:
            TargetProperties('regression', regression_type='l2')
        }

        model = MultiHeadedOneTokenPerCellModel(
            target_to_properties=target_to_properties,
            model_size=self.model_size,
            dropout_rate=0.1).to(self.device)
        model.load_state_dict(state_dict, strict=True)

        _, eval_preds, _ = compute_predictions(
            model,
            X,
            'test',
            as_probabilities=False,
            target_column=self.target_column,
            device=self.device,
            is_classification=self.is_classification,
            id_mappings=id_mappings,
            processed_dataset_cache=CACHE_PATH)
        eval_preds = scaler.inverse_transform(
            np.array(eval_preds).reshape(-1, 1)).flatten()
        return np.asarray(eval_preds)

    def predict(self, X):
        masked_df = pd.concat([X, pd.DataFrame({self.target_column: [0.0 for _ in range(len(X))]}, index=X.index)],
                              axis=1)
        preds = self._predict(masked_df, False)
        return pd.Series(preds, name=self.target_column, index=X.index)
