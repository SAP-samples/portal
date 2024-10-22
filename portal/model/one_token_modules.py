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
    create_tenant_mask,
    embedding_model_to_dimension_and_pooling,
    masked_argmax,
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


class TripletHead(SSLHead):
    def __init__(self, config, is_cosine_similarity=False):
        super().__init__(config, ['triplet'], [config.hidden_size], [])
        self.is_cosine_similarity = is_cosine_similarity

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1).sqrt()

    def forward(self, features: torch.Tensor, loss_mask: torch.Tensor, triplet_classes=None, is_validation=False):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        logits = {k: layer(x) for k, layer in self.layers.items()}

        if is_validation:
            # return the embedding for the input row
            return logits, None
        else:
            logits_triplet = logits['triplet']
            batch_split = logits_triplet.shape[0] // 3
            anchor = logits_triplet[:batch_split, :]
            positive = logits_triplet[batch_split:-batch_split, :]
            negative = logits_triplet[-batch_split:, :]

            if triplet_classes is None:
                if self.is_cosine_similarity:
                    current_margin = 0.1
                else:
                    current_margin = 1
            else:
                # include distance between percentile classes. Far away percentile classes should be separated by larger margin
                if self.is_cosine_similarity:
                    current_margin = 0.1 + torch.abs(triplet_classes[:, 1] - triplet_classes[:, 0]) / 200.0
                else:
                    current_margin = 1.0 + torch.abs(triplet_classes[:, 1] - triplet_classes[:, 0]) / 10.0

            # triplet_loss = functional.triplet_margin_loss(anchor, positive, negative, margin=1, reduction='none')
            if self.is_cosine_similarity:
                distance_positive = 1 - functional.cosine_similarity(anchor, positive)
                distance_negative = 1 - functional.cosine_similarity(anchor, negative)
            else:
                distance_positive = self.calc_euclidean(anchor, positive)
                distance_negative = self.calc_euclidean(anchor, negative)

            triplet_loss = torch.relu(distance_positive - distance_negative + current_margin)

            triplet_loss = (triplet_loss * loss_mask[:, 0]).sum() / (loss_mask[:, 0] + 1e-5).sum()
            return {'triplet': anchor}, triplet_loss


class RobertaSSL(nn.Module, ModuleUtilsMixin):
    def __init__(self,
                 sentence_embedding_model_name: str,
                 model_size: ModelSize,
                 use_number_percentiles=True,
                 is_triplet=False,
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
        self.is_triplet = is_triplet
        if self.is_triplet:
            self.triplet_head = TripletHead(self.config)
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.config.hidden_size))

    def forward(self,
                input_ids: Dict,
                labels: Optional[Dict] = None,
                attention_mask: Optional[torch.Tensor] = None,
                loss_mask: Optional[torch.Tensor] = None,
                anchor_ids: Optional[torch.Tensor] = None,
                positive_ids: Optional[Dict] = None,
                negative_ids: Optional[Dict] = None,
                triplet_loss_mask: Optional[torch.Tensor] = None,
                is_validation=False,
                **kwargs):
        device = input_ids['is_positive'].device
        input_size = input_ids['is_positive'].size()
        if attention_mask is None:
            attention_mask = torch.ones(input_size, device=device)
        input_embeds = self.embeddings(input_ids)

        if self.is_triplet:
            # case for validation
            if is_validation:
                anchor_embeds = self.embeddings(labels)

                # extend mask for CLS token
                attention_mask_ext = torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)
                attention_mask = torch.column_stack((attention_mask_ext, attention_mask))
                attention_mask = torch.cat([attention_mask, attention_mask])

                cls_token = self.cls_token.repeat(input_embeds.shape[0], 1, 1)
                input_embeds = torch.column_stack((cls_token, input_embeds))  # tokens is of shape [B, 1+T, F]
                anchor_embeds = torch.column_stack((cls_token, anchor_embeds))  # tokens is of shape [B, 1+T, F]

                # we pass through transformer input_ids and anchor
                input_embeds = torch.cat([input_embeds, anchor_embeds])
                input_size = torch.tensor([input_size[0] * 2, input_size[1] + 1])
            else:
                anchor_embeds = self.embeddings(anchor_ids)
                positive_embeds = self.embeddings(positive_ids)
                negative_embeds = self.embeddings(negative_ids)

                # extend mask for CLS token
                attention_mask_ext = torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)
                attention_mask = torch.column_stack((attention_mask_ext, attention_mask))
                attention_mask = torch.cat([attention_mask, attention_mask, attention_mask, attention_mask])

                cls_token = self.cls_token.repeat(input_embeds.shape[0], 1, 1)
                input_embeds = torch.column_stack((cls_token, input_embeds))  # tokens is of shape [B, 1+T, F]
                anchor_embeds = torch.column_stack((cls_token, anchor_embeds))  # tokens is of shape [B, 1+T, F]
                positive_embeds = torch.column_stack((cls_token, positive_embeds))  # tokens is of shape [B, 1+T, F]
                negative_embeds = torch.column_stack((cls_token, negative_embeds))  # tokens is of shape [B, 1+T, F]

                input_embeds = torch.cat([input_embeds, anchor_embeds, positive_embeds, negative_embeds])
                input_size = torch.tensor([input_size[0] * 4, input_size[1] + 1])

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_size)
        encoder_outputs = self.encoder(input_embeds, attention_mask=extended_attention_mask, return_dict=False)

        if self.is_triplet:
            encoder_output_without_cls = encoder_outputs[0][:, 1:, :]
            # case for validation
            if is_validation:
                # skip input_embeds, get anchor
                encoder_output_cls = encoder_outputs[0][input_embeds.shape[0] // 2:, 0, :]
                encoder_outputs = encoder_output_without_cls[:input_embeds.shape[0] // 2, :, :]
                encoder_outputs = (encoder_outputs, )
                attention_mask = attention_mask[:attention_mask.shape[0] // 2, 1:]
            else:
                # skip input_embeds, get anchor, positive, negative
                encoder_output_cls = encoder_outputs[0][input_embeds.shape[0] // 4:, 0, :]
                encoder_outputs = encoder_output_without_cls[:input_embeds.shape[0] // 4, :, :]
                encoder_outputs = (encoder_outputs, )
                attention_mask = attention_mask[:attention_mask.shape[0] // 4, 1:]

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

        if self.is_triplet:
            if is_validation:
                logits_triplet, loss_triplet = self.triplet_head(encoder_output_cls, None, is_validation=is_validation)
            else:
                logits_triplet, loss_triplet = self.triplet_head(encoder_output_cls,
                                                                 triplet_loss_mask,
                                                                 is_validation=is_validation)
            logits.update(logits_triplet)
        else:
            loss_triplet = torch.tensor(float('nan'))

        logits.update(logits_number)
        logits.update(logits_date)

        result = {'logits': logits, 'attentions': encoder_outputs[1:]}

        if labels is None or loss_mask is None or is_number is None or is_date is None or is_text is None:
            return result
        assert number_mask is not None and date_mask is not None and text_mask is not None  # Just to silence VisualStudio errors

        # Text loss is computed below
        result['loss'] = {'number': loss_number, 'date': loss_date, 'text': loss_text, 'triplet': loss_triplet}
        result['counts'] = {
            'number': is_number.sum(),
            'date': is_date.sum(),
            'text': is_text.sum(),
            'loss': loss_mask.sum(),
            'loss_number': number_mask.sum(),
            'loss_date': date_mask.sum(),
            'loss_text': text_mask.sum(),
            'loss_triplet': loss_triplet,
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

    def forward(self, input_ids, labels=None, attention_mask=None, tenant_id=None, **kwargs):
        device = input_ids['is_positive'].device
        input_size = input_ids['is_positive'].size()

        if attention_mask is None:
            attention_mask = torch.ones(input_size, device=device)

        input_embeds = self.embeddings(input_ids)

        return self.forward_post_embeddings(input_embeds, attention_mask, input_size, labels, tenant_id=tenant_id)


class MultiHeadedOneTokenPerCellTripletModel(MultiHeadedOneTokenPerCellModel):
    def __init__(self,
                 target_to_properties: Dict[str, TargetProperties],
                 model_size: ModelSize,
                 tokenizer_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 use_number_percentiles: bool = False,
                 weighted_loss_dict: Optional[Dict[str, Dict[int, float]]] = None,
                 regression_as_classification=False,
                 is_cosine_similarity=False,
                 **kwargs):
        self.regression_as_classification = regression_as_classification
        self.is_cosine_similarity = is_cosine_similarity
        super().__init__(target_to_properties, model_size, tokenizer_model_name, use_number_percentiles,
                         weighted_loss_dict)

    def create_heads(self):
        self.triplet_head = TripletHead(self.config, is_cosine_similarity=self.is_cosine_similarity)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.config.hidden_size))

    def forward(self,
                input_ids,
                positive_ids=None,
                negative_ids=None,
                triplet_loss_mask=None,
                triplet_classes=None,
                labels=None,
                attention_mask=None,
                tenant_id=None,
                is_validation=False,
                **kwargs):
        device = input_ids['is_positive'].device
        input_size = input_ids['is_positive'].size()

        # we have extra CLS token
        if is_validation:
            input_size = (input_size[0], input_size[1] + 1)
        else:
            # we pass input, positive, negative at once
            input_size = (3 * input_size[0], input_size[1] + 1)

        if attention_mask is None:
            attention_mask = torch.ones(input_size, device=device)

        input_embeds = self.embeddings(input_ids)

        positive_embeds, negative_embeds = None, None
        if is_validation == False:
            positive_embeds = self.embeddings(positive_ids)
            negative_embeds = self.embeddings(negative_ids)

        return self.forward_post_embeddings(input_embeds,
                                            attention_mask,
                                            input_size,
                                            labels,
                                            positive_embeds,
                                            negative_embeds,
                                            triplet_loss_mask,
                                            triplet_classes,
                                            is_validation=is_validation)

    def forward_post_embeddings(self,
                                input_embeds,
                                attention_mask,
                                input_size,
                                labels,
                                positive_embeds=None,
                                negative_embeds=None,
                                triplet_loss_mask=None,
                                triplet_classes=None,
                                is_validation=False):
        cls_token = self.cls_token.repeat(input_embeds.shape[0], 1, 1)
        input_embeds = torch.column_stack((cls_token, input_embeds))  # tokens is of shape [B, 1+T, F]
        if is_validation == False:
            positive_embeds = torch.column_stack((cls_token, positive_embeds))  # tokens is of shape [B, 1+T, F]
            negative_embeds = torch.column_stack((cls_token, negative_embeds))  # tokens is of shape [B, 1+T, F]
            input_embeds = torch.cat([input_embeds, positive_embeds, negative_embeds])

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_size)
        encoder_outputs = self.encoder(input_embeds, attention_mask=extended_attention_mask, return_dict=False)
        sequence_output = encoder_outputs[0]

        sequence_output_cls = sequence_output[:, 0, :]
        if not self.regression_as_classification:
            # consider triplet classes for margin calculation only if we use percentiles in the regression task
            triplet_classes = None
        logits, loss = self.triplet_head(sequence_output_cls,
                                         triplet_loss_mask,
                                         triplet_classes=triplet_classes,
                                         is_validation=is_validation)

        output = (logits, ) + encoder_outputs[1:]
        return ((loss, ) + output) if loss is not None else output


class OneTokenPerCellLikeSSLModel(RobertaSSL):
    # This wrapper modifies the __init__ to be compatible with MultiHeadedOneTokenPerCellModel
    # However, outputs will _not_ be compatible, this needs to be taken care of by the lightning module
    def __init__(self,
                 target_to_properties: Dict[str, TargetProperties],
                 model_size: ModelSize,
                 tokenizer_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 use_number_percentiles: bool = False,
                 regression_type: str = 'mixed',
                 **kwargs):
        # This model doesn't care about target_to_properties, since it will always
        # make a prediction per each token, both the targets and the features.
        super().__init__(sentence_embedding_model_name=tokenizer_model_name,
                         use_number_percentiles=use_number_percentiles,
                         model_size=model_size,
                         regression_type=regression_type)

        self.target_to_properties = target_to_properties
        self.regression_type = regression_type
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=2)

    def extract_classification_predictions(self,
                                           logits: Dict[str, torch.Tensor],
                                           target_properties: TargetProperties,
                                           tenant_id: Optional[List[str]] = None):
        text_logits = logits['text']
        # Shape: (batch_size, EMBEDDING_DIMENSION)
        assert text_logits.ndim == 2
        # Shape: (num_classes, EMBEDDING_DIMENSION)
        assert target_properties.class_embeddings is not None and target_properties.class_embeddings.ndim == 2, str(
            target_properties.class_embeddings)

        similarities = self.cosine_similarity(text_logits.detach().float().cpu().unsqueeze(1),
                                              target_properties.class_embeddings.unsqueeze(0))
        if tenant_id:
            mask = create_tenant_mask(shape=similarities.shape,
                                      values_per_tenant=target_properties.values_per_tenant,
                                      tenant_id=tenant_id)
            argmax = partial(masked_argmax, mask=mask)
        else:
            argmax = torch.argmax

        return argmax(similarities, dim=1)

    def extract_predictions(self,
                            logits: Dict[str, torch.Tensor],
                            tenant_id: Optional[List[str]] = None,
                            as_probabilities=False):
        predictions = {}

        column_name_to_id = {str(name): i for i, name in enumerate(logits['column_names'])}

        for head_name, target_properties in self.target_to_properties.items():
            if head_name not in column_name_to_id:
                # This column was not present in the input data (probably constant and thus dropped)
                continue
            this_id = column_name_to_id[head_name]
            these_logits = {k: v[:, this_id] for k, v in logits.items() if k != 'column_names'}

            if target_properties.type == 'classification':
                predictions[head_name] = self.extract_classification_predictions(these_logits,
                                                                                 target_properties,
                                                                                 tenant_id=tenant_id)
                if as_probabilities:
                    # This method is not able to predict probabilities (at least, without calibration).
                    # We just return probability 1 for the most likely class.
                    preds = torch.zeros(these_logits['text'].shape[0], target_properties.size, dtype=torch.float32)
                    preds[torch.arange(preds.shape[0]), predictions[head_name]] = 1.0
            else:
                number_mask = torch.ones(these_logits['is_positive'].shape[0], dtype=torch.bool)
                predictions[head_name] = self.extract_regression_predictions(these_logits, number_mask)

        return predictions
