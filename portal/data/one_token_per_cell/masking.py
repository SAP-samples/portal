import sys
import random
import torch
import numpy as np
import pandas as pd



def _mask_tensor(tensor, k):
    # Masking cells will have NaN as number and NaT as time:
    if k in [
            'is_positive', 'fraction_bin', 'delta', 'date_year', 'date_month', 'date_day', 'date_weekday',
            'date_holidays', 'content_embeddings'
    ]:
        return torch.zeros_like(tensor)
    if k == 'exponent':
        return torch.zeros_like(tensor) + 255
    if k == 'percentile_values':
        return torch.zeros_like(tensor) + 101

    raise ValueError(f'key {k} not understood in masking!')


def mask_one_row(input_ids, do_mask, other_ids=None, do_replace=None):
    for k in input_ids:
        # Column names are never masked; content embeddings are treated like
        # any other value (masked by zeros, or replaced by other embeddings)
        if k not in ['column_embeddings', 'is_number', 'is_date']:
            input_ids[k][do_mask] = _mask_tensor(input_ids[k][0], k)
            if other_ids is not None:
                input_ids[k][do_replace] = other_ids[k][do_replace]
    return input_ids


def get_triplet_random_masks(num_tokens, possible_columns):
    if possible_columns is not None:
        true_idx = np.argwhere(possible_columns)
        random_idx = random.choice(true_idx)[0]
    else:
        random_idx = random.randint(0, num_tokens - 1)

    mask = torch.zeros(num_tokens, dtype=bool)
    mask[random_idx] = True

    return mask, np.zeros_like(mask), ~mask


def get_random_masks(num_tokens, mlm_probability):
    weights = torch.tensor([0.8 * mlm_probability, 0.1 * mlm_probability, 0.1 * mlm_probability, 1 - mlm_probability])
    random_numbers = torch.multinomial(weights, num_tokens, replacement=True)

    do_mask = random_numbers == 0
    do_replace = random_numbers == 1
    # The unchanged ones are... well, unchanged, we don't need them
    # do_unchanged = random_numbers == 2
    do_no_loss = random_numbers == 3
    if num_tokens > 1 and do_no_loss.all():
        # If we have two columns or more, at least one should
        # contribute to the loss, otherwise why are we even
        # wasting time on this sample?
        return get_random_masks(num_tokens, mlm_probability)
    return do_mask, do_replace, do_no_loss


def do_random_masking(mlm_probability, input_ids, other_ids, is_triplet=False, possible_columns=None):
    num_tokens = len(input_ids['is_positive'])
    labels = {k: torch.clone(v) for k, v in input_ids.items()}
    do_mask, do_replace, do_no_loss = get_random_masks(num_tokens, mlm_probability)

    triplet_do_no_loss = torch.ones(num_tokens) == True
    if is_triplet:
        _, _, triplet_do_no_loss = get_triplet_random_masks(num_tokens, possible_columns)

    input_ids = mask_one_row(input_ids, do_mask, other_ids, do_replace)
    return input_ids, labels, ~do_no_loss, ~triplet_do_no_loss
