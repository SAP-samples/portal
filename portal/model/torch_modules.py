import re
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Optional

import torch
from torch import nn
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead, RobertaConfig, RobertaEncoder

from portal.constants import UNKNOWN_CLASS_LABEL_ID, ModelSize
from portal.data.data_utils import (
    EXPONENT_BITS,
    decode_numbers_torch,
    decode_numbers_torch_no_binning,
    encode_numbers_torch,
    encode_numbers_torch_no_binning,
)

embedding_model_to_dimension_and_pooling = {
    'sentence-transformers/all-MiniLM-L6-v2': (384, 'mean'),
    'BAAI/bge-large-en-v1.5': (1024, 'cls'),
    'BAAI/bge-base-en-v1.5': (768, 'cls'),
    'BAAI/bge-small-en-v1.5': (384, 'cls'),
    'thenlper/gte-large': (1024, 'mean'),
    'thenlper/gte-base': (768, 'mean'),
    'thenlper/gte-small': (384, 'mean'),
    'intfloat/e5-base-v2': (768, 'mean')
}


class NumberEmbeddings(nn.Module):
    def __init__(self, hidden_size, use_number_percentiles):
        super().__init__()
        self.is_positive_embeddings = nn.Embedding(2, hidden_size)
        self.exponent_embeddings = nn.Embedding(256, hidden_size)
        self.fraction_bin_embeddings = nn.Embedding(1001, hidden_size)
        if use_number_percentiles:
            self.percentile_embeddings = nn.Embedding(102, hidden_size)
        else:
            self.percentile_embeddings = None

    def forward(self, is_positive, exponent, fraction_bin, delta, percentile_values=None):
        is_positive_embeds = self.is_positive_embeddings(is_positive)
        exponent_embeds = self.exponent_embeddings(exponent)
        if self.percentile_embeddings is None:
            percentile_embeds = 0.0
        else:
            assert percentile_values is not None, 'percentile_values must be provided if use_number_percentiles is True'
            percentile_embeds = self.percentile_embeddings(percentile_values)
        # add empty dimension to delta: (batch_size, length) -> (batch_size, length, 1)
        # since it needs to broadcast with (batch_size, length, hidden_size)
        delta = torch.unsqueeze(delta, 2)
        fraction_bin_embeds = self.fraction_bin_embeddings(fraction_bin) * (
            1.0 - delta) + self.fraction_bin_embeddings(fraction_bin + 1) * delta
        return (is_positive_embeds + exponent_embeds + fraction_bin_embeds + percentile_embeds).type(
            is_positive_embeds.dtype)


class DateEmbeddings(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.year_embeddings = nn.Embedding(51, hidden_size)
        self.month_embeddings = nn.Embedding(13, hidden_size)
        self.day_embeddings = nn.Embedding(32, hidden_size)
        self.weekday_embeddings = nn.Embedding(8, hidden_size)
        self.holidays_embeddings = nn.Linear(120, hidden_size)

    def forward(self, year, month, day, weekday, holidays):
        year_embeds = self.year_embeddings(year)
        month_embeds = self.month_embeddings(month)
        day_embeds = self.day_embeddings(day)
        weekday_embeds = self.weekday_embeddings(weekday)
        holidays_embeds = self.holidays_embeddings(holidays.type(self.holidays_embeddings.weight.dtype))
        return year_embeds + month_embeds + day_embeds + weekday_embeds + holidays_embeds


def logits_to_scientific(logits: torch.Tensor):
    exponent_logits = logits[:, :EXPONENT_BITS]
    fraction_logits = logits[:, EXPONENT_BITS:-1]
    is_positive_logit = logits[:, -1]
    return exponent_logits, fraction_logits, is_positive_logit


class MultiHeadedModel(ABC, nn.Module, ModuleUtilsMixin):
    """
    Pytorch Model for multiple tasks with RoBERTa for tokenized input.
    Each head can be either a classification or a regression head.
    """
    @abstractmethod
    def __init__(self,
                 target_to_properties: Dict[str, 'portal.utils.target_properties.TargetProperties'],
                 model_size: ModelSize,
                 weighted_loss_dict: Optional[Dict[str, Dict[int, float]]] = None):
        super().__init__()
        self.target_to_properties = target_to_properties
        self.weighted_loss_dict = weighted_loss_dict
        self.model_size = model_size.value
        num_layers, hidden_size = model_size.value
        self.config = RobertaConfig(num_hidden_layers=num_layers,
                                    hidden_size=hidden_size,
                                    num_attention_heads=hidden_size // 64,
                                    layer_norm_eps=1e-5,
                                    type_vocab_size=1)

        # Here we define the encoder and the heads for each target
        # The first embedding layer(s) will have to be defined by the concrete classes.
        self.encoder = RobertaEncoder(self.config)

    @abstractmethod
    def create_embeddings(self):
        ...

    def forward_post_embeddings(self, input_embeds, attention_mask, input_size, labels):
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_size)
        encoder_outputs = self.encoder(input_embeds, attention_mask=extended_attention_mask, return_dict=False)
        sequence_output = encoder_outputs[0]

        logits = {
            target_name: self.heads[self.sanitize_name(target_name)](sequence_output)
            for target_name in self.target_to_properties
        }
        mask_per_target = None
        loss = self.compute_loss(labels, logits, mask_per_target=mask_per_target)

        output = (logits, ) + encoder_outputs[1:]
        return ((loss, ) + output) if loss is not None else output

    @staticmethod
    def sanitize_name(name):
        # Module names cannot contain "." character. Better safe than sorry, let's
        # replace any non-alphanumeric character by "_"
        return re.sub(r'\W', '_', name)

    def create_heads(self):
        heads = dict()
        for target_name, target_properties in self.target_to_properties.items():
            self.config.num_labels = target_properties.size
            # We use RobertaClassificationHead even for regression - in that case,
            # the head size is 1 or whatever is needed by the specific regression_type we use
            heads[self.sanitize_name(target_name)] = RobertaClassificationHead(self.config)
        self.heads = nn.ModuleDict(heads)

    def compute_classification_loss(self,
                                    labels: torch.Tensor,
                                    logits: torch.Tensor,
                                    weight: Optional[torch.Tensor],
                                    mask: Optional[torch.Tensor] = None):
        device = logits.device
        batch_size = labels.numel()

        # We explicitly convert labels to long, because if there are multiple heads, one of which of
        # regression type, the labels have been put together into a tensor and implicitly converted to float.
        logits = logits.view(batch_size, -1).float().to(device)
        labels = labels.view(batch_size).to(device).long()
        weight = weight.to(device) if weight is not None else None
        mask = mask.view(batch_size, -1).to(device) if mask is not None else None

        if mask is not None:
            mask = mask.float()
            # zero out the masked elements in logspace
            # 1 + 1e-42 is essentially 0 and 0 + 1e-42 is something very low - close enough 8-)
            # https://github.com/allenai/allennlp/blob/b6cc9d39651273e8ec2a7e334908ffa9de5c2026/allennlp/nn/util.py#L272-L303
            logits = logits + torch.log(mask + 1e-42)

        return nn.functional.nll_loss(
            nn.functional.log_softmax(logits, dim=-1), labels, weight=weight,
            ignore_index=UNKNOWN_CLASS_LABEL_ID)  # unknown labels in val/test set get mapped to -100

    @staticmethod
    def compute_cross_entropy_loss_exponent_and_is_positive(exponent: torch.Tensor, exponent_logits: torch.Tensor,
                                                            is_positive: torch.Tensor, is_positive_logit: torch.Tensor):
        # exponent_logits: size (N, EXPONENT_BITS)
        # exponent: size (N,), values between 0 and EXPONENT_BITS already:
        loss = nn.functional.cross_entropy(exponent_logits.float(), exponent)

        # is_positive_logit: size (N,) type float
        # is_positive: size (N,) type bool -> convert to int64
        loss += nn.functional.binary_cross_entropy_with_logits(is_positive_logit.float(), is_positive.float())

        return loss

    def compute_cross_entropy_loss(self, labels: torch.Tensor, logits: torch.Tensor):
        # labels is a torch tensor; the following code converts the result to numpy arrays
        # (is_positive of type bool, delta is float32, exponent and fraction_bin are int64)
        is_positive, exponent, fraction_bin, delta = encode_numbers_torch(labels)
        exponent_logits, fraction_logits, is_positive_logit = logits_to_scientific(logits)

        loss = self.compute_cross_entropy_loss_exponent_and_is_positive(exponent, exponent_logits, is_positive,
                                                                        is_positive_logit)

        # Half precision loss is apparently not implemented? Cast to float.
        fraction_logits = fraction_logits.float()

        # fraction_logits: size (N, FRACTION_BINS + 1)
        # fraction_bin: size (N,) type int64
        # delta: size (N,) type float32
        # We need to reconstruct soft labels having 0 everywhere
        # except at indices fraction_bin and fraction_bin + 1
        fraction_labels = torch.zeros_like(fraction_logits)
        N = fraction_labels.size(0)
        delta = delta.float().type(fraction_logits.dtype)
        fraction_labels[torch.arange(N), fraction_bin] = 1 - delta
        fraction_labels[torch.arange(N), fraction_bin + 1] = delta
        loss += nn.functional.cross_entropy(fraction_logits, fraction_labels)

        return loss

    def compute_mixed_loss(self, labels: torch.Tensor, logits: torch.Tensor):
        # labels is a torch tensor; the following code converts the result to numpy arrays
        # (is_positive of type bool, fraction is float32 and in [0, 1], exponent is int64)
        is_positive, exponent, fraction = encode_numbers_torch_no_binning(labels)
        exponent_logits, fraction_logits, is_positive_logit = logits_to_scientific(logits)

        loss = self.compute_cross_entropy_loss_exponent_and_is_positive(exponent, exponent_logits, is_positive,
                                                                        is_positive_logit)

        pred_fraction = fraction_logits.squeeze(-1).float()
        return loss + nn.functional.binary_cross_entropy_with_logits(pred_fraction, fraction)

    def compute_l2_loss(self, labels: torch.Tensor, logits: torch.Tensor):
        # logits: size (N, 1) --> needs squeeze
        # labels: size (N,) type float64
        return nn.functional.mse_loss(logits.squeeze(-1), labels)

    def compute_loss(self,
                     labels: Optional[dict],
                     logits: dict,
                     mask_per_target: Optional[Dict[str, torch.Tensor]] = None):
        if not labels:
            return None

        loss = 0.0

        for head, target_properties in self.target_to_properties.items():
            if head in labels:
                if target_properties.type == 'classification':
                    loss += self.compute_classification_loss(labels[head], logits[head], self.get_weights(head),
                                                             mask_per_target[head] if mask_per_target else None)
                elif target_properties.regression_type == 'cross_entropy':
                    loss += self.compute_cross_entropy_loss(labels[head], logits[head])
                elif target_properties.regression_type == 'l2':
                    loss += self.compute_l2_loss(labels[head], logits[head])
                else:
                    loss += self.compute_mixed_loss(labels[head], logits[head])

        return loss / len(self.target_to_properties)

    def get_weights(self, target):
        if self.weighted_loss_dict:
            head_size = self.target_to_properties[target].size
            assert head_size is not None, 'get_weights cannot be used for regression'
            #Ensure the weights are in the correct order based on class labels
            weights = torch.tensor([(self.weighted_loss_dict.get(target, defaultdict(lambda: 1))).get(label, 1)
                                    for label in range(head_size)],
                                   dtype=torch.float32).to('cpu')
            return weights
        else:
            return None

    def extract_predictions(self,
                            logits: Dict[str, torch.Tensor],
                            as_probabilities=False):
        """
        logits: dict[str, Tensor], with Tensor of shape (batch_size, d) where d depends on the task:
            number of classes for classification, 1 for l2 regression, currently 257 for mixed regression,
            and 1257 for cross_entropy.
        """
        predictions = {}

        for head_name, target_properties in self.target_to_properties.items():
            if target_properties.type == 'classification':
                if as_probabilities:
                    predictions[head_name] = torch.softmax(logits[head_name], dim=1)
                else:
                    predictions[head_name] = torch.argmax(logits[head_name], dim=1)
            else:
                if target_properties.regression_type == 'l2':
                    predictions[head_name] = logits[head_name].squeeze(-1).float()
                else:
                    exponent_logits, fraction_logits, is_positive_logit = logits_to_scientific(logits[head_name])
                    exponent_pred = torch.argmax(exponent_logits, dim=1)
                    is_positive = is_positive_logit > 0
                    if target_properties.regression_type == 'mixed':
                        fraction01_preds = torch.sigmoid(fraction_logits.squeeze(1))
                        predictions[head_name] = decode_numbers_torch_no_binning(is_positive, exponent_pred,
                                                                                 fraction01_preds.squeeze(-1))
                    else:
                        # This is in [0, FRACTION_BINS]:
                        fraction_bin = torch.argmax(fraction_logits, dim=1)
                        # TODO: implement delta, i.e. look at the two neighboring bins of fraction_bin,
                        # choose the biggest neighbour (or the only one if fraction_bin is an extreme),
                        # take softmax of these two values, and define delta accordingly.
                        predictions[head_name] = decode_numbers_torch(is_positive, exponent_pred, fraction_bin)
        return predictions
