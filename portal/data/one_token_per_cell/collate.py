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


def collate_fn(samples):
    result = {}
    result['input_ids'] = pad_list_of_dict([sample['input_ids'] for sample in samples])
    if samples[0].get('triplet_loss_mask') is not None:
        result['anchor_ids'] = pad_list_of_dict([sample['anchor_ids'] for sample in samples])
        result['positive_ids'] = pad_list_of_dict([sample['positive_ids'] for sample in samples])
        result['negative_ids'] = pad_list_of_dict([sample['negative_ids'] for sample in samples])
        result['triplet_loss_mask'] = pad_sequence([sample['triplet_loss_mask'] for sample in samples], batch_first=True)

    result['labels'] = pad_list_of_dict([sample['labels'] for sample in samples])
    result['attention_mask'] = pad_sequence([torch.ones_like(sample['loss_mask']) for sample in samples],
                                            batch_first=True)
    result['loss_mask'] = pad_sequence([sample['loss_mask'] for sample in samples], batch_first=True)

    if 'filename' in samples[0]:
        result['filename'] = [sample['filename'] for sample in samples]
    if 'row_index' in samples[0]:
        result['row_index'] = [sample['row_index'] for sample in samples]

    if 'other_values' in samples[0]:
        # This happens only in validation
        result['other_values'] = pad_other_values([sample['other_values'] for sample in samples])
    return result
