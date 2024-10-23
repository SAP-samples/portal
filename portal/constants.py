import argparse
import os
from enum import Enum
from pathlib import Path
from typing import Final


CACHE_PATH = Path(os.environ['HOME'], '.cache', 'portal')
CACHE_PATH.mkdir(parents=True, exist_ok=True)
print('CACHE_PATH:', CACHE_PATH)

DEFAULT_ZMQ_PORT: Final[int] = 5555

UNKNOWN_CLASS_LABEL_ID: Final[int] = -100


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
