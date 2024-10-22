from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Set, Union

from portal.data.data_utils import EXPONENT_BITS, FRACTION_BINS

if TYPE_CHECKING:
    # Avoid cyclic import, since this is used only in typing
    from portal.data.one_token_per_cell.row_tokenizer import StringTokenizer


class TargetProperties:
    """
    Class that stores basic properties of the target column. The main property is:
    - type: classification or regression
    For classification:
    - classes: for classification: list of class names, or just their number;
        if it's a list, it's saved in class_list, and also class_embeddings is populated if string_tokenizer is provided
        In that case, class_embeddings is a tensor.
    - regression_type must be None
    For regression:
    - classes: should be None, but it can also be the integer number of targets.
    - regression_type: cross_entropy, mixed, l2
    - string_tokenizer should be None (ignored anyway)
    """
    def __init__(self,
                 type: Literal['classification', 'regression'],
                 classes: Optional[Union[int, list]] = None,
                 regression_type: Optional[Literal['cross_entropy', 'mixed', 'l2']] = None,
                 string_tokenizer: Optional['StringTokenizer'] = None,
                 values_per_tenant: Optional[Dict[str, Set[int]]] = None):
        self.type = type
        self.regression_type = regression_type
        self.values_per_tenant = values_per_tenant  # valid label values per tenant
        self.class_list = None
        self.class_embeddings = None

        if self.type == 'classification':
            assert classes is not None, 'size must be provided for classification head'
            assert self.regression_type is None, 'regression_type must be None for classification head'
            if isinstance(classes, int):
                size = classes
            else:
                size = len(classes)
                self.class_list = list(classes)
                if string_tokenizer is not None:
                    class_list_str = []
                    for cl in self.class_list:
                        if not isinstance(cl, str):
                            print(
                                'Warning, class', cl, 'is not a string. '
                                'Classification will still work, but be less accurate if this is a numeric type.')
                            cl = str(cl)
                        class_list_str.append(cl)
                    self.class_embeddings = string_tokenizer.texts_to_tensor(class_list_str).float()
                    # Awful hack to make sure the sockets are shut down.
                    # This is needed with multiprocessing because this one method is called from the dataset,
                    # while data is loaded in the dataloader, and leaving this one open will somehow cause
                    # all the other ones to hang indefinitely. No idea why.
                    string_tokenizer.cleanup_socket()
        elif self.type == 'regression':
            if self.regression_type == 'cross_entropy':
                # We need FRACTION_BINS + 1 classes for the fraction part,
                # because if e.g. FRACTION_BINS = 1000, then fraction_bin is in 0..999,
                # but when delta > 0 we also need one more bin to covert the range [1.999, 2.0)
                target_size = EXPONENT_BITS + FRACTION_BINS + 2
            elif self.regression_type == 'mixed':
                target_size = EXPONENT_BITS + 2
            elif self.regression_type == 'l2':
                target_size = 1
            else:
                raise ValueError(f'Unknown regression type {self.regression_type}')

            # sanity check (when loading from trained model, size must match)
            if classes is None:
                size = target_size
            else:
                assert isinstance(classes, int), 'size must be None or an integer for regression head'
                size = classes
                assert size == target_size, f'size of {self.regression_type} regression head mismatch: passed={size}, expected={target_size}'
        else:
            raise ValueError(f'Unknown head type {self.type}')

        assert isinstance(size, int)
        self.size: int = size

    def __repr__(self):
        return f'TargetProperties(type={self.type}, classes={self.class_list or self.size}, regression_type={self.regression_type}, computed class_embeddings={self.class_embeddings is not None})'

    @classmethod
    def from_dict(cls, d: dict):
        if 'size' in d:
            if 'classes' in d:
                print('Warning: both size (legacy) and classes (new) parameters are provided, ignoring size, but why?')
                del d['size']
            else:
                d['classes'] = d.pop('size')
        return cls(**d)

    def to_dict(self):
        return {
            'type': self.type,
            'size': self.size,
            'regression_type': self.regression_type,
            'values_per_tenant':
            {k: list(v)
             for k, v in self.values_per_tenant.items()} if self.values_per_tenant is not None else None
        }
