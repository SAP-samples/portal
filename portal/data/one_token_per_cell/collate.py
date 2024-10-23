from math import nan

import torch
from torch.nn.utils.rnn import pad_sequence


def pad_list_of_dict(list_of_dict):
    """
    Fields in dictionary have the following shape:
    - most of them are 1-dimensional, with one element per table column (l,)
    - date_holidays has shape (l, 120)
    - column_embeddings and content_embeddings have shape (l, embedding_dim)
        e.g. embedding_size for all-MiniLM-L6-v2 is 384
    We have to batch them all together, by padding `l` to the max number
    of columns in any element of the batch, and also padding m and n to
    their max values.
    (both m and n are currently capped at 128)
    """
    result = {}

    keys = list(list_of_dict[0])

    # For embeddings, padding is done with zeros.
    # This is the same as unknown, actually, which might be not great, but hopefully we don't get
    # any unknown at all?
    # In training, we should have computed all the embeddings already, so we should never get unknown;
    # in inference, embeddings should be computed on the fly.

    for key in keys:
        result[key] = pad_sequence([dictionary[key] for dictionary in list_of_dict], batch_first=True)

    return result


def pad_other_values(other_values):
    return {
        # Each value of dates and numbers are 1d tensors of numbers
        # (length: number_of_rows of the original dataframe they were in)
        # We pad them with nan, which is unused in validation anyway
        'dates': pad_sequence([value['date'] for value in other_values], padding_value=nan, batch_first=True),
        'numbers': pad_sequence([value['number'] for value in other_values], padding_value=nan, batch_first=True),
        # content_embeddings instead have two dimensions (number_of_rows, embedding_dim)
        # but still pad_sequence works for them
        'content_embeddings': pad_sequence([value['content_embeddings'] for value in other_values], batch_first=True)
    }
