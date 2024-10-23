from functools import partial
from math import log10
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.nn import functional
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.models.roberta.modeling_roberta import RobertaConfig, RobertaEncoder, gelu

from portal.constants import ModelSize
from portal.data.data_utils import decode_numbers, decode_numbers_torch_no_binning, encode_numbers_torch_no_binning
from portal.model.torch_modules import (
    DateEmbeddings,
    MultiHeadedModel,
    NumberEmbeddings,
    embedding_model_to_dimension_and_pooling,
)
from portal.utils.target_properties import TargetProperties


class OneTokenEmbeddings(nn.Module):
    """
    Embedding module for self supervised learning.
    On the input side, it sums four contributions:
    - Numbers (itself coming from embedding three one-hot encoded values: sign, exponent, fraction)
    - Dates (itself coming from embedding four one-hot encoded values: year, month, day, weekday, plus one multi-hot encoded: holidays)
    - Column names (sentence embedding of the column name, adjusted to the hidden size)
    - (String) contents (sentence embedding of the column name, adjusted to the hidden size)
    For labels, it also computes and returns the sentence embedding of the string contents (not adjusted to the size).
    All string embeddings (column names, contents of both input and labels)
    """
    def __init__(self, config, sentence_embedding_model_name: str, use_number_percentiles: bool):
        super().__init__()
        self.number_embeddings = NumberEmbeddings(config.hidden_size, use_number_percentiles=use_number_percentiles)
        self.date_embeddings = DateEmbeddings(config.hidden_size)
        self.embedding_dimension, _ = embedding_model_to_dimension_and_pooling[sentence_embedding_model_name]
        # Remark: it's important that the two following layers don't share weights,
        # otherwise there would be no way to distinguish between column "A" having
        # content "B" and column "B" having content "A".
        self.column_remapping = nn.Linear(self.embedding_dimension, config.hidden_size)
        self.content_remapping = nn.Linear(self.embedding_dimension, config.hidden_size)

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_dict: Dict):
        number_embeds = self.number_embeddings(input_dict['is_positive'], input_dict['exponent'],
                                               input_dict['fraction_bin'], input_dict['delta'],
                                               input_dict.get('percentile_values'))
        date_embeds = self.date_embeddings(input_dict['date_year'], input_dict['date_month'], input_dict['date_day'],
                                           input_dict['date_weekday'], input_dict['date_holidays'])

        is_number = input_dict['is_number']
        is_date = input_dict['is_date']
        is_text = ~(is_number | is_date)
        is_number = is_number.unsqueeze(-1).float()
        is_date = is_date.unsqueeze(-1).float()
        is_text = is_text.unsqueeze(-1).float()
        number_embeds = number_embeds * is_number
        date_embeds = date_embeds * is_date

        # column_embeddings and content_embeddings have shape (batch_size, max_column_number, EMBEDDING_DIMENSION)
        # The following remaps it to (batch_size, max_column_number, HIDDEN_SIZE), same as the other embeddings.
        column_embeds = self.column_remapping(input_dict['column_embeddings'].type(self.column_remapping.weight.dtype))
        content_embeds = self.content_remapping(input_dict['content_embeddings'].type(
            self.content_remapping.weight.dtype))

        content_embeds = content_embeds * is_text
        input_embeds = column_embeds + content_embeds + number_embeds + date_embeds
        input_embeds = self.layer_norm(input_embeds)
        input_embeds = self.dropout(input_embeds)
        return input_embeds


class SSLHead(nn.Module):
    def __init__(self,
                 config,
                 layer_names: List[str],
                 layer_num_labels: List[int],
                 binary_layers: List[str],
                 regression_type='cross_entropy'):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        layers = {}
        for key, num_labels in zip(layer_names, layer_num_labels):
            layers[key] = nn.Linear(config.hidden_size, num_labels)
        self.layers = nn.ModuleDict(layers)
        self.binary_layers = binary_layers
        self.regression_type = regression_type

    def compute_loss(self, logits: dict, labels: dict, loss_mask: torch.Tensor):
        if self.regression_type == 'mixed':
            labels_decoded = decode_numbers(labels['is_positive'], labels['exponent'], labels['fraction_bin'],
                                            labels['delta'])
            labels['is_positive'], labels['exponent'], labels['fraction_bin'] = encode_numbers_torch_no_binning(
                labels_decoded)
            labels['exponent'] = labels['exponent'].long()

        loss = 0.0
        for key in self.layers:
            if key in self.binary_layers:
                pred = logits[key].view(-1, logits[key].size(-1))
                gt = labels[key].view(pred.size()).float()
                loss += functional.binary_cross_entropy_with_logits(pred, gt, reduction='none').mean(dim=-1)
            else:
                loss += functional.cross_entropy(logits[key].view(-1, logits[key].size(-1)).float(),
                                                 labels[key].view(-1),
                                                 reduction='none')
        loss_mask = loss_mask.view(-1).float()
        loss *= loss_mask
        return loss.sum() / (loss_mask + 1e-5).sum()

    def forward(self, features: torch.Tensor, labels: Optional[Dict] = None, loss_mask: Optional[torch.Tensor] = None):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        logits = {k: layer(x) for k, layer in self.layers.items()}

        if labels is None or loss_mask is None:
            return logits, None

        loss = self.compute_loss(logits, labels, loss_mask)

        return logits, loss


class TextHead(SSLHead):
    def __init__(self, config, sentence_embedding_model_name):
        super().__init__(config, ['text'], [embedding_model_to_dimension_and_pooling[sentence_embedding_model_name][0]],
                         [])

    def compute_loss(self, logits: dict, labels: torch.Tensor, loss_mask: torch.Tensor):
        # Make sure dtype is the same; labels here are typically float16
        labels = labels.type(logits['text'].dtype)
        text_loss = functional.huber_loss(logits['text'], labels, reduction='none')
        # text_loss has shape (batch_size, column_number, SENTENCE_EMBEDDING_DIMENSION)
        # loss_mask has shape (batch_size, column_number)
        loss_mask = loss_mask.unsqueeze(-1).float()
        loss = (text_loss * loss_mask).sum() / (loss_mask + 1e-5).sum()
        return loss


class RobertaSSL(nn.Module, ModuleUtilsMixin):
    def __init__(self,
                 sentence_embedding_model_name: str,
                 model_size: ModelSize,
                 use_number_percentiles=True,
                 regression_type='cross_entropy',
                 dropout_rate=0.1):
        super().__init__()
        num_layers, hidden_size = model_size.value
        self.config = RobertaConfig(num_hidden_layers=num_layers,
                                    hidden_size=hidden_size,
                                    num_attention_heads=hidden_size // 64,
                                    layer_norm_eps=1e-5,
                                    hidden_dropout_prob=dropout_rate)
        self.regression_type = regression_type

        self.embeddings = OneTokenEmbeddings(self.config,
                                             sentence_embedding_model_name,
                                             use_number_percentiles=use_number_percentiles)
        self.encoder = RobertaEncoder(self.config)
        number_of_fraction_bins = 1001
        binary_layers = ['is_positive']
        if regression_type == 'mixed':
            number_of_fraction_bins = 1
            binary_layers.append('fraction_bin')

        self.number_head = SSLHead(self.config, ['is_positive', 'exponent', 'fraction_bin'],
                                   [1, 256, number_of_fraction_bins],
                                   binary_layers,
                                   regression_type=regression_type)
        self.date_head = SSLHead(self.config, ['date_year', 'date_month', 'date_day'], [51, 13, 32], [])
        self.text_head = TextHead(self.config, sentence_embedding_model_name)

    def forward(self,
                input_ids: Dict,
                labels: Optional[Dict] = None,
                attention_mask: Optional[torch.Tensor] = None,
                loss_mask: Optional[torch.Tensor] = None,
                anchor_ids: Optional[torch.Tensor] = None,
                positive_ids: Optional[Dict] = None,
                negative_ids: Optional[Dict] = None,
                is_validation=False,
                **kwargs):
        device = input_ids['is_positive'].device
        input_size = input_ids['is_positive'].size()
        if attention_mask is None:
            attention_mask = torch.ones(input_size, device=device)
        input_embeds = self.embeddings(input_ids)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_size)
        encoder_outputs = self.encoder(input_embeds, attention_mask=extended_attention_mask, return_dict=False)

        number_mask = date_mask = text_mask = is_text = None
        is_number, is_date = input_ids['is_number'], input_ids['is_date']
        if is_number is not None and is_date is not None:
            is_text = ~is_number & ~is_date

        if loss_mask is not None:
            number_mask = loss_mask & is_number
            date_mask = loss_mask & is_date
            text_mask = loss_mask & is_text
        logits_number, loss_number = self.number_head(encoder_outputs[0], labels, number_mask)
        logits_date, loss_date = self.date_head(encoder_outputs[0], labels, date_mask)
        label_embeddings = labels['content_embeddings'] if labels is not None else None
        logits, loss_text = self.text_head(encoder_outputs[0], label_embeddings, text_mask)

        logits.update(logits_number)
        logits.update(logits_date)

        result = {'logits': logits, 'attentions': encoder_outputs[1:]}

        if labels is None or loss_mask is None or is_number is None or is_date is None or is_text is None:
            return result
        assert number_mask is not None and date_mask is not None and text_mask is not None  # Just to silence VisualStudio errors

        # Text loss is computed below
        result['loss'] = {'number': loss_number, 'date': loss_date, 'text': loss_text}
        result['counts'] = {
            'number': is_number.sum(),
            'date': is_date.sum(),
            'text': is_text.sum(),
            'loss': loss_mask.sum(),
            'loss_number': number_mask.sum(),
            'loss_date': date_mask.sum(),
            'loss_text': text_mask.sum(),
        }

        return result

    def extract_regression_predictions(self, logits: Dict[str, torch.Tensor], number_mask: torch.Tensor):
        is_positive = (logits['is_positive'][number_mask].squeeze(-1) > 0).float()
        exponent = torch.argmax(logits['exponent'][number_mask], dim=-1)

        if self.regression_type == 'mixed':
            fraction01_preds = torch.sigmoid(logits['fraction_bin'][number_mask]).squeeze(1)
            pred_numbers = decode_numbers_torch_no_binning(is_positive, exponent, fraction01_preds)
        else:
            fraction_bin = torch.argmax(logits['fraction_bin'][number_mask], dim=-1)
            pred_numbers = decode_numbers(is_positive, exponent, fraction_bin)
        return pred_numbers


class MultiHeadedOneTokenPerCellModel(MultiHeadedModel):
    def __init__(self,
                 target_to_properties: Dict[str, TargetProperties],
                 model_size: ModelSize,
                 tokenizer_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 use_number_percentiles: bool = False,
                 weighted_loss_dict: Optional[Dict[str, Dict[int, float]]] = None,
                 **kwargs):
        super().__init__(target_to_properties, model_size, weighted_loss_dict)

        self.create_embeddings(tokenizer_model_name, use_number_percentiles)
        self.create_heads()

    def create_embeddings(self, tokenizer_model_name, use_number_percentiles: bool):
        self.embeddings = OneTokenEmbeddings(self.config,
                                             sentence_embedding_model_name=tokenizer_model_name,
                                             use_number_percentiles=use_number_percentiles)

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        device = input_ids['is_positive'].device
        input_size = input_ids['is_positive'].size()

        if attention_mask is None:
            attention_mask = torch.ones(input_size, device=device)

        input_embeds = self.embeddings(input_ids)

        return self.forward_post_embeddings(input_embeds, attention_mask, input_size, labels)
