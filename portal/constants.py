import argparse
import getpass
import os
import re
import subprocess
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Final

import pandas as pd



CACHE_PATH = Path(os.environ['HOME'], '.cache', 'portal')
CACHE_PATH.mkdir(parents=True, exist_ok=True)
print('CACHE_PATH:', CACHE_PATH)


DEFAULT_ZMQ_PORT: Final[int] = 5555

BEST_CKPT_FILENAME: Final[str] = 'best.ckpt'
UNKNOWN_CLASS_LABEL_ID: Final[int] = -100
TENANT_ID_COL = '__tenant_id__'


OPEN_FINE_TUNE_EXPERIMENT_NAME = {
    'carte': 'OpenFineTuneResultsCarte',
    '50k_subsample': 'OpenFineTuneResults50k',
    'vime': 'OpenFineTuneResultsVime',
}


class SplitName:
    TRAIN: Final[str] = 'train'
    VALIDATION: Final[str] = 'validation'
    TEST: Final[str] = 'test'


class ModelType(Enum):
    DUMMY = 'dummy'  # Used for unit tests
    ONE_TOKEN_PER_CELL = 'one-token-per-cell'
    MANY_TOKENS_PER_CELL = 'many-tokens-per-cell'
    ONE_TOKEN_PER_CELL_LIKE_SSL = 'one-token-per-cell-like-ssl'
    ONE_TOKEN_PER_CELL_TRIPLET = 'one-token-per-cell-triplet'


class ModelTypeAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        try:
            # Search by value:
            value = ModelType(values)
        except ValueError:
            # Search by key:
            if not isinstance(values, str):
                raise ValueError(
                    f'{values} has type {type(values)} which is not string, but this does not match any of the possible values {ModelType.__members__.values()}'
                )
            try:
                value = ModelType[values]
            except KeyError:
                possible_dict = {k: v.value for k, v in ModelType.__members__.items()}
                raise ValueError(f'{values} is neither a value nor a key for ModelType: {possible_dict}')
        setattr(namespace, self.dest, value)


class ModelSize(Enum):
    # The two values are the number of layers and the hidden size
    tiny = (2, 128)
    mini = (4, 256)
    small = (4, 512)
    medium = (8, 512)
    base = (12, 768)
    large = (24, 1024)
    xlarge = (24, 2048)


class ModelSizeAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values not in ModelSize.__members__ or not isinstance(values, str):
            raise ValueError(f'{values} is not a valid value for ModelSize: {ModelSize.__members__.keys()}')
        value = ModelSize[values]
        setattr(namespace, self.dest, value)
