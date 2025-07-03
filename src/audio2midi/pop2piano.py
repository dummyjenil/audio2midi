import torch
import copy
import os
import numpy as np
import pretty_midi_fix
from librosa.core import resample as librosa_resample
from scipy.interpolate import interp1d
from json import load as json_load , dumps as json_dumps
from math import log as math_log
from typing import Optional, Union
from torch import nn
from librosa import load as librosa_load
from huggingface_hub import snapshot_download
from essentia.standard import RhythmExtractor2013
from transformers.generation import GenerationConfig , GenerationMixin
from transformers.activations import ACT2FN
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput
from transformers.utils import is_torch_flex_attn_available, is_torch_fx_proxy, is_torchdynamo_compiling , TensorType, to_numpy
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.configuration_utils import PretrainedConfig
from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.audio_utils import mel_filter_bank, spectrogram
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.feature_extraction_utils import BatchFeature
from transformers.tokenization_utils import AddedToken, BatchEncoding, PaddingStrategy, PreTrainedTokenizer, TruncationStrategy

if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask
    from transformers.integrations.flex_attention import make_flex_block_causal_mask


class Pop2PianoConfig(PretrainedConfig):
    model_type = "pop2piano"
    keys_to_ignore_at_inference = ["past_key_values"]
    def __init__(
        self,
        vocab_size=2400,
        composer_vocab_size=21,
        d_model=512,
        d_kv=64,
        d_ff=2048,
        num_layers=6,
        num_decoder_layers=None,
        num_heads=8,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="gated-gelu",  # noqa
        is_encoder_decoder=True,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        dense_act_fn="relu",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.composer_vocab_size = composer_vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = num_decoder_layers if num_decoder_layers is not None else self.num_layers
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.use_cache = use_cache
        self.dense_act_fn = dense_act_fn
        self.is_gated_act = self.feed_forward_proj.split("-")[0] == "gated"
        self.hidden_size = self.d_model
        self.num_attention_heads = num_heads
        self.num_hidden_layers = num_layers

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )


# Copied from transformers.models.t5.modeling_t5.T5LayerNorm with T5->Pop2Piano
class Pop2PianoLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the Pop2Piano style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # Pop2Piano uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://huggingface.co/papers/1910.07467 thus variance is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states

# from apex.normalization import FusedRMSNorm
# Pop2PianoLayerNorm = FusedRMSNorm  # noqa
# # Other Approach

# Copied from transformers.models.t5.modeling_t5.T5DenseActDense with T5->Pop2Piano,t5->pop2piano
class Pop2PianoDenseActDense(nn.Module):
    def __init__(self, config: Pop2PianoConfig):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states


# Copied from transformers.models.t5.modeling_t5.T5DenseGatedActDense with T5->Pop2Piano
class Pop2PianoDenseGatedActDense(nn.Module):
    def __init__(self, config: Pop2PianoConfig):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)

        # To make 8bit quantization work for google/flan-t5-xxl, self.wo is kept in float32.
        # See https://github.com/huggingface/transformers/issues/20287
        # we also make sure the weights are not in `int8` in case users will force `_keep_in_fp32_modules` to be `None``
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)

        hidden_states = self.wo(hidden_states)
        return hidden_states


# Copied from transformers.models.t5.modeling_t5.T5LayerFF with T5->Pop2Piano
class Pop2PianoLayerFF(nn.Module):
    def __init__(self, config: Pop2PianoConfig):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = Pop2PianoDenseGatedActDense(config)
        else:
            self.DenseReluDense = Pop2PianoDenseActDense(config)

        self.layer_norm = Pop2PianoLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


# Copied from transformers.models.t5.modeling_t5.T5Attention with T5->Pop2Piano,t5->pop2piano
class Pop2PianoAttention(nn.Module):
    def __init__(
        self,
        config: Pop2PianoConfig,
        has_relative_attention_bias=False,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.layer_idx = layer_idx
        if layer_idx is None and self.is_decoder:
            print(
                f"Instantiating a decoder {self.__class__.__name__} without passing `layer_idx` is not recommended and "
                "will to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()
        self.gradient_checkpointing = False

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math_log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None, cache_position=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        if cache_position is None:
            context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        else:
            context_position = cache_position[:, None].to(device)
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
        cache_position=None,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, 1, 1, key_length) (non-causal encoder) or (batch_size, 1, seq_length, key_length) (causal decoder)
        batch_size, seq_length = hidden_states.shape[:2]

        # if key_value_states are provided this layer is used as a cross-attention layer for the decoder
        is_cross_attention = key_value_states is not None

        query_states = self.q(hidden_states)
        query_states = query_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                curr_past_key_value = past_key_value.cross_attention_cache
            else:
                curr_past_key_value = past_key_value.self_attention_cache

        current_states = key_value_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_value is not None and is_updated:
            # reuse k,v, cross_attentions
            key_states = curr_past_key_value.key_cache[self.layer_idx]
            value_states = curr_past_key_value.value_cache[self.layer_idx]
        else:
            key_states = self.k(current_states)
            value_states = self.v(current_states)
            key_states = key_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
            value_states = value_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

            if past_key_value is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = curr_past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )
                # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                if is_cross_attention:
                    past_key_value.is_updated[self.layer_idx] = True

        # compute scores, equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
        scores = torch.matmul(query_states, key_states.transpose(3, 2))

        if position_bias is None:
            key_length = key_states.shape[-2]
            # cache position is 0-indexed so we add 1 to get the real length of queries (aka with past)
            real_seq_length = query_length if query_length is not None else cache_position[-1] + 1
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(
                    real_seq_length, key_length, device=scores.device, cache_position=cache_position
                )
                position_bias = position_bias[:, :, -seq_length:, :]

            if mask is not None:
                causal_mask = mask[:, :, :, : key_states.shape[-2]]
                position_bias = position_bias + causal_mask

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        scores += position_bias_masked

        # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.inner_dim)
        attn_output = self.o(attn_output)

        outputs = (attn_output, past_key_value, position_bias)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


# Copied from transformers.models.t5.modeling_t5.T5LayerSelfAttention with T5->Pop2Piano,t5->pop2piano
class Pop2PianoLayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False, layer_idx: Optional[int] = None):
        super().__init__()
        self.SelfAttention = Pop2PianoAttention(
            config, has_relative_attention_bias=has_relative_attention_bias, layer_idx=layer_idx
        )
        self.layer_norm = Pop2PianoLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        cache_position=None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.t5.modeling_t5.T5LayerCrossAttention with T5->Pop2Piano,t5->pop2piano
class Pop2PianoLayerCrossAttention(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.EncDecAttention = Pop2PianoAttention(config, has_relative_attention_bias=False, layer_idx=layer_idx)
        self.layer_norm = Pop2PianoLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
        cache_position=None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs

# Copied from transformers.models.t5.modeling_t5.T5Block with T5->Pop2Piano,t5->pop2piano
class Pop2PianoBlock(GradientCheckpointingLayer):
    def __init__(self, config, has_relative_attention_bias=False, layer_idx: Optional[int] = None):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(
            Pop2PianoLayerSelfAttention(
                config, has_relative_attention_bias=has_relative_attention_bias, layer_idx=layer_idx
            )
        )
        if self.is_decoder:
            self.layer.append(Pop2PianoLayerCrossAttention(config, layer_idx=layer_idx))

        self.layer.append(Pop2PianoLayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
        cache_position=None,
    ):
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        hidden_states, past_key_value = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                query_length=cache_position[-1] + 1,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states, past_key_value = cross_attention_outputs[:2]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.where(
                    torch.isinf(hidden_states).any(),
                    torch.finfo(hidden_states.dtype).max - 1000,
                    torch.finfo(hidden_states.dtype).max,
                )
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (past_key_value,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, past_key_value, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)

class Pop2PianoConcatEmbeddingToMel(nn.Module):
    """Embedding Matrix for `composer` tokens."""

    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=config.composer_vocab_size, embedding_dim=config.d_model)

    def forward(self, feature, index_value, embedding_offset):
        index_shifted = index_value - embedding_offset
        composer_embedding = self.embedding(index_shifted).unsqueeze(1)
        inputs_embeds = torch.cat([composer_embedding, feature], dim=1)
        return inputs_embeds

class Pop2PianoPreTrainedModel(PreTrainedModel):
    config_class = Pop2PianoConfig
    base_model_prefix = "transformer"
    is_parallelizable = False
    supports_gradient_checkpointing = True
    _supports_cache_class = True
    _supports_static_cache = False
    _no_split_modules = ["Pop2PianoBlock"]
    _keep_in_fp32_modules = ["wo"]

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, Pop2PianoLayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, Pop2PianoConcatEmbeddingToMel):
            module.embedding.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, Pop2PianoForConditionalGeneration):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, "lm_head") and not self.config.tie_word_embeddings:
                module.lm_head.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, Pop2PianoDenseActDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, Pop2PianoDenseGatedActDense):
            module.wi_0.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_0, "bias") and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            module.wi_1.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_1, "bias") and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, Pop2PianoAttention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In Pop2Piano it is usually set to the pad_token_id."
            )

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

class Pop2PianoStack(Pop2PianoPreTrainedModel):
    # Copied from transformers.models.t5.modeling_t5.T5Stack.__init__ with T5->Pop2Piano,t5->pop2piano
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [
                Pop2PianoBlock(config, has_relative_attention_bias=bool(i == 0), layer_idx=i)
                for i in range(config.num_layers)
            ]
        )
        self.final_layer_norm = Pop2PianoLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    # Copied from transformers.models.t5.modeling_t5.T5Stack.get_input_embeddings
    def get_input_embeddings(self):
        return self.embed_tokens

    # Copied from transformers.models.t5.modeling_t5.T5Stack.set_input_embeddings
    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                print(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError("You have to initialize the model with valid token embeddings")
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")

        # initialize past_key_values
        return_legacy_cache = False
        return_self_attention_cache = False
        if self.is_decoder and (use_cache or past_key_values is not None):
            if isinstance(past_key_values, Cache) and not isinstance(past_key_values, EncoderDecoderCache):
                return_self_attention_cache = True
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())
            elif not isinstance(past_key_values, EncoderDecoderCache):
                return_legacy_cache = True
                print(
                    "Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.48.0. "
                    "You should pass an instance of `EncoderDecoderCache` instead, e.g. "
                    "`past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values)`."
                )
                past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values)
            elif past_key_values is None:
                past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
        elif not self.is_decoder:
            # do not pass cache object down the line for encoder stack
            # it messes indexing later in decoder-stack because cache object is modified in-place
            past_key_values = None

        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length, past_key_values_length + seq_length, device=inputs_embeds.device
            )

        if attention_mask is None and not is_torchdynamo_compiling():
            # required mask seq length can be calculated via length of past cache
            mask_seq_length = past_key_values_length + seq_length
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)

        if self.config.is_decoder:
            causal_mask = self._update_causal_mask(
                attention_mask,
                inputs_embeds,
                cache_position,
                past_key_values.self_attention_cache if past_key_values is not None else None,
                output_attentions,
            )
        else:
            causal_mask = attention_mask[:, None, None, :]
            causal_mask = causal_mask.to(dtype=inputs_embeds.dtype)
            causal_mask = (1.0 - causal_mask) * torch.finfo(inputs_embeds.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, layer_module in enumerate(self.block):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                causal_mask,
                position_bias,
                encoder_hidden_states,
                encoder_extended_attention_mask,
                encoder_decoder_position_bias,  # as a positional argument for gradient checkpointing
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                cache_position=cache_position,
            )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, next_decoder_cache = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_self_attention_cache:
            next_cache = past_key_values.self_attention_cache
        if return_legacy_cache:
            next_cache = past_key_values.to_legacy_cache()

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )

    # Copied from transformers.models.gptj.modeling_gptj.GPTJModel._update_causal_mask
    def _update_causal_mask(
        self,
        attention_mask: Union[torch.Tensor, "BlockMask"],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_compilable_cache = past_key_values.is_compileable if past_key_values is not None else False

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_compilable_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype = input_tensor.dtype
        sequence_length = input_tensor.shape[1]
        if using_compilable_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    # Copied from transformers.models.gptj.modeling_gptj.GPTJModel._prepare_4d_causal_attention_mask_with_cache_position
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask

class Pop2PianoForConditionalGeneration(Pop2PianoPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: Pop2PianoConfig):
        super().__init__(config)
        self.config = config
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        self.mel_conditioner = Pop2PianoConcatEmbeddingToMel(config)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        self.encoder = Pop2PianoStack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = Pop2PianoStack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_mel_conditioner_outputs(
        self,
        input_features: torch.FloatTensor,
        composer: str,
        generation_config: GenerationConfig,
        attention_mask: Optional[torch.FloatTensor] = None,
    ):
        """
        This method is used to concatenate mel conditioner tokens at the front of the input_features in order to
        control the type of MIDI token generated by the model.

        Args:
            input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                input features extracted from the feature extractor.
            composer (`str`):
                composer token which determines the type of MIDI tokens to be generated.
            generation_config (`~generation.GenerationConfig`):
                The generation is used to get the composer-feature_token pair.
            attention_mask (``, *optional*):
                For batched generation `input_features` are padded to have the same shape across all examples.
                `attention_mask` helps to determine which areas were padded and which were not.
                - 1 for tokens that are **not padded**,
                - 0 for tokens that are **padded**.
        """
        composer_to_feature_token = generation_config.composer_to_feature_token
        if composer not in composer_to_feature_token.keys():
            raise ValueError(
                f"Please choose a composer from {list(composer_to_feature_token.keys())}. Composer received - {composer}"
            )
        composer_value = composer_to_feature_token[composer]
        composer_value = torch.tensor(composer_value, device=self.device)
        composer_value = composer_value.repeat(input_features.shape[0])

        embedding_offset = min(composer_to_feature_token.values())

        input_features = self.mel_conditioner(
            feature=input_features,
            index_value=composer_value,
            embedding_offset=embedding_offset,
        )
        if attention_mask is not None:
            input_features[~attention_mask[:, 0].bool()] = 0.0

            # since self.mel_conditioner adds a new array at the front of inputs_embeds we need to do the same for attention_mask to keep the shapes same
            attention_mask = torch.concatenate([attention_mask[:, 0].view(-1, 1), attention_mask], axis=1)
            return input_features, attention_mask

        return input_features, None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[tuple[tuple[torch.Tensor]]] = None,
        past_key_values: Optional[tuple[tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Pop2Piano is a model with relative position embeddings
            so you should be able to pad the inputs on both the right and the left. Indices can be obtained using
            [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for detail.
            [What are input IDs?](../glossary#input-ids) To know more on how to prepare `input_ids` for pretraining
            take a look a [Pop2Piano Training](./Pop2Piano#training).
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary. Indices can be obtained using
            [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.
            [What are decoder input IDs?](../glossary#decoder-input-ids) Pop2Piano uses the `pad_token_id` as the
            starting token for `decoder_input_ids` generation. If `past_key_values` is used, optionally only the last
            `decoder_input_ids` have to be input (see `past_key_values`). To know more on how to prepare
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        decoder_head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in `[0,
            1]`:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        cross_attn_head_mask (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
            `[0, 1]`:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Does the same task as `inputs_embeds`. If `inputs_embeds` is not present but `input_features` is present
            then `input_features` will be considered as `inputs_embeds`.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is not None and input_features is not None:
            raise ValueError("Both `inputs_embeds` and `input_features` received! Please provide only one of them")
        elif input_features is not None and inputs_embeds is None:
            inputs_embeds = input_features

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        input_features,
        attention_mask=None,
        composer="composer1",
        generation_config=None,
        **kwargs,
    ):
        """
        Generates token ids for midi outputs.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`. For an overview of generation
        strategies and code examples, check out the [following guide](./generation_strategies).

        </Tip>

        Parameters:
            input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                This is the featurized version of audio generated by `Pop2PianoFeatureExtractor`.
            attention_mask:
                For batched generation `input_features` are padded to have the same shape across all examples.
                `attention_mask` helps to determine which areas were padded and which were not.
                - 1 for tokens that are **not padded**,
                - 0 for tokens that are **padded**.
            composer (`str`, *optional*, defaults to `"composer1"`):
                This value is passed to `Pop2PianoConcatEmbeddingToMel` to generate different embeddings for each
                `"composer"`. Please make sure that the composet value is present in `composer_to_feature_token` in
                `generation_config`. For an example please see
                https://huggingface.co/sweetcocoa/pop2piano/blob/main/generation_config.json .
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            kwargs:
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.
        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.
                Since Pop2Piano is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:
                    - [`~generation.GenerateEncoderDecoderOutput`],
                    - [`~generation.GenerateBeamEncoderDecoderOutput`]
        """

        if generation_config is None:
            generation_config = self.generation_config
        generation_config.update(**kwargs)

        # check for composer_to_feature_token
        if not hasattr(generation_config, "composer_to_feature_token"):
            raise ValueError(
                "`composer_to_feature_token` was not found! Please refer to "
                "https://huggingface.co/sweetcocoa/pop2piano/blob/main/generation_config.json"
                "and parse a dict like that."
            )

        if len(generation_config.composer_to_feature_token) != self.config.composer_vocab_size:
            raise ValueError(
                "config.composer_vocab_size must be same as the number of keys in "
                f"generation_config.composer_to_feature_token! "
                f"Found {self.config.composer_vocab_size} vs {len(generation_config.composer_to_feature_token)}."
            )

        # to control the variation of generated MIDI tokens we concatenate mel-conditioner tokens(which depends on composer_token)
        # at the front of input_features.
        input_features, attention_mask = self.get_mel_conditioner_outputs(
            input_features=input_features,
            attention_mask=attention_mask,
            composer=composer,
            generation_config=generation_config,
        )

        return super().generate(
            inputs=None,
            inputs_embeds=input_features,
            attention_mask=attention_mask,
            generation_config=generation_config,
            **kwargs,
        )

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past_key_values, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past_key_values is None:
            print("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(
                    f"reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched"
                )
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(
                    f"length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched"
                )

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

class Pop2PianoFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a Pop2Piano feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    This class extracts rhythm and preprocesses the audio before it is passed to the model. First the audio is passed
    to `RhythmExtractor2013` algorithm which extracts the beat_times, beat positions and estimates their confidence as
    well as tempo in bpm, then beat_times is interpolated and to get beatsteps. Later we calculate
    extrapolated_beatsteps from it to be used in tokenizer. On the other hand audio is resampled to self.sampling_rate
    and preprocessed and then log mel spectogram is computed from that to be used in our transformer model.

    Args:
        sampling_rate (`int`, *optional*, defaults to 22050):
            Target Sampling rate of audio signal. It's the sampling rate that we forward to the model.
        padding_value (`int`, *optional*, defaults to 0):
            Padding value used to pad the audio. Should correspond to silences.
        window_size (`int`, *optional*, defaults to 4096):
            Length of the window in samples to which the Fourier transform is applied.
        hop_length (`int`, *optional*, defaults to 1024):
            Step size between each window of the waveform, in samples.
        min_frequency (`float`, *optional*, defaults to 10.0):
            Lowest frequency that will be used in the log-mel spectrogram.
        feature_size (`int`, *optional*, defaults to 512):
            The feature dimension of the extracted features.
        num_bars (`int`, *optional*, defaults to 2):
            Determines interval between each sequence.
    """

    model_input_names = ["input_features", "beatsteps", "extrapolated_beatstep"]

    def __init__(
        self,
        sampling_rate: int = 22050,
        padding_value: int = 0,
        window_size: int = 4096,
        hop_length: int = 1024,
        min_frequency: float = 10.0,
        feature_size: int = 512,
        num_bars: int = 2,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value
        self.window_size = window_size
        self.hop_length = hop_length
        self.min_frequency = min_frequency
        self.feature_size = feature_size
        self.num_bars = num_bars
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=(self.window_size // 2) + 1,
            num_mel_filters=self.feature_size,
            min_frequency=self.min_frequency,
            max_frequency=float(self.sampling_rate // 2),
            sampling_rate=self.sampling_rate,
            norm=None,
            mel_scale="htk",
        )

    def mel_spectrogram(self, sequence: np.ndarray):
        """
        Generates MelSpectrogram.

        Args:
            sequence (`np.ndarray`):
                The sequence of which the mel-spectrogram will be computed.
        """
        mel_specs = []
        for seq in sequence:
            window = np.hanning(self.window_size + 1)[:-1]
            mel_specs.append(
                spectrogram(
                    waveform=seq,
                    window=window,
                    frame_length=self.window_size,
                    hop_length=self.hop_length,
                    power=2.0,
                    mel_filters=self.mel_filters,
                )
            )
        mel_specs = np.array(mel_specs)

        return mel_specs

    def extract_rhythm(self, audio: np.ndarray):
        """
        This algorithm(`RhythmExtractor2013`) extracts the beat positions and estimates their confidence as well as
        tempo in bpm for an audio signal. For more information please visit
        https://essentia.upf.edu/reference/std_RhythmExtractor2013.html .

        Args:
            audio(`np.ndarray`):
                raw audio waveform which is passed to the Rhythm Extractor.
        """
        essentia_tracker = RhythmExtractor2013(method="multifeature")
        bpm, beat_times, confidence, estimates, essentia_beat_intervals = essentia_tracker(audio)

        return bpm, beat_times, confidence, estimates, essentia_beat_intervals

    def interpolate_beat_times(
        self, beat_times: np.ndarray, steps_per_beat: np.ndarray, n_extend: np.ndarray
    ):
        """
        This method takes beat_times and then interpolates that using `scipy.interpolate.interp1d` and the output is
        then used to convert raw audio to log-mel-spectrogram.

        Args:
            beat_times (`np.ndarray`):
                beat_times is passed into `scipy.interpolate.interp1d` for processing.
            steps_per_beat (`int`):
                used as an parameter to control the interpolation.
            n_extend (`int`):
                used as an parameter to control the interpolation.
        """

        beat_times_function = interp1d(
            np.arange(beat_times.size),
            beat_times,
            bounds_error=False,
            fill_value="extrapolate",
        )

        ext_beats = beat_times_function(
            np.linspace(0, beat_times.size + n_extend - 1, beat_times.size * steps_per_beat + n_extend)
        )

        return ext_beats

    def preprocess_mel(self, audio: np.ndarray, beatstep: np.ndarray):
        """
        Preprocessing for log-mel-spectrogram

        Args:
            audio (`np.ndarray` of shape `(audio_length, )` ):
                Raw audio waveform to be processed.
            beatstep (`np.ndarray`):
                Interpolated values of the raw audio. If beatstep[0] is greater than 0.0, then it will be shifted by
                the value at beatstep[0].
        """

        if audio is not None and len(audio.shape) != 1:
            raise ValueError(
                f"Expected `audio` to be a single channel audio input of shape `(n, )` but found shape {audio.shape}."
            )
        if beatstep[0] > 0.0:
            beatstep = beatstep - beatstep[0]

        num_steps = self.num_bars * 4
        num_target_steps = len(beatstep)
        extrapolated_beatstep = self.interpolate_beat_times(
            beat_times=beatstep, steps_per_beat=1, n_extend=(self.num_bars + 1) * 4 + 1
        )

        sample_indices = []
        max_feature_length = 0
        for i in range(0, num_target_steps, num_steps):
            start_idx = i
            end_idx = min(i + num_steps, num_target_steps)
            start_sample = int(extrapolated_beatstep[start_idx] * self.sampling_rate)
            end_sample = int(extrapolated_beatstep[end_idx] * self.sampling_rate)
            sample_indices.append((start_sample, end_sample))
            max_feature_length = max(max_feature_length, end_sample - start_sample)
        padded_batch = []
        for start_sample, end_sample in sample_indices:
            feature = audio[start_sample:end_sample]
            padded_feature = np.pad(
                feature,
                ((0, max_feature_length - feature.shape[0]),),
                "constant",
                constant_values=0,
            )
            padded_batch.append(padded_feature)

        padded_batch = np.asarray(padded_batch)
        return padded_batch, extrapolated_beatstep

    def _pad(self, features: np.ndarray, add_zero_line=True):
        features_shapes = [each_feature.shape for each_feature in features]
        attention_masks, padded_features = [], []
        for i, each_feature in enumerate(features):
            # To pad "input_features".
            if len(each_feature.shape) == 3:
                features_pad_value = max([*zip(*features_shapes)][1]) - features_shapes[i][1]
                attention_mask = np.ones(features_shapes[i][:2], dtype=np.int64)
                feature_padding = ((0, 0), (0, features_pad_value), (0, 0))
                attention_mask_padding = (feature_padding[0], feature_padding[1])

            # To pad "beatsteps" and "extrapolated_beatstep".
            else:
                each_feature = each_feature.reshape(1, -1)
                features_pad_value = max([*zip(*features_shapes)][0]) - features_shapes[i][0]
                attention_mask = np.ones(features_shapes[i], dtype=np.int64).reshape(1, -1)
                feature_padding = attention_mask_padding = ((0, 0), (0, features_pad_value))

            each_padded_feature = np.pad(each_feature, feature_padding, "constant", constant_values=self.padding_value)
            attention_mask = np.pad(
                attention_mask, attention_mask_padding, "constant", constant_values=self.padding_value
            )

            if add_zero_line:
                # if it is batched then we separate each examples using zero array
                zero_array_len = max([*zip(*features_shapes)][1])

                # we concatenate the zero array line here
                each_padded_feature = np.concatenate(
                    [each_padded_feature, np.zeros([1, zero_array_len, self.feature_size])], axis=0
                )
                attention_mask = np.concatenate(
                    [attention_mask, np.zeros([1, zero_array_len], dtype=attention_mask.dtype)], axis=0
                )

            padded_features.append(each_padded_feature)
            attention_masks.append(attention_mask)

        padded_features = np.concatenate(padded_features, axis=0).astype(np.float32)
        attention_masks = np.concatenate(attention_masks, axis=0).astype(np.int64)

        return padded_features, attention_masks

    def pad(
        self,
        inputs: BatchFeature,
        is_batched: bool,
        return_attention_mask: bool,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ):
        """
        Pads the inputs to same length and returns attention_mask.

        Args:
            inputs (`BatchFeature`):
                Processed audio features.
            is_batched (`bool`):
                Whether inputs are batched or not.
            return_attention_mask (`bool`):
                Whether to return attention mask or not.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
                If nothing is specified, it will return list of `np.ndarray` arrays.
        Return:
            `BatchFeature` with attention_mask, attention_mask_beatsteps and attention_mask_extrapolated_beatstep added
            to it:
            - **attention_mask** np.ndarray of shape `(batch_size, max_input_features_seq_length)` --
                Example :
                    1, 1, 1, 0, 0 (audio 1, also here it is padded to max length of 5 that's why there are 2 zeros at
                    the end indicating they are padded)

                    0, 0, 0, 0, 0 (zero pad to separate audio 1 and 2)

                    1, 1, 1, 1, 1 (audio 2)

                    0, 0, 0, 0, 0 (zero pad to separate audio 2 and 3)

                    1, 1, 1, 1, 1 (audio 3)
            - **attention_mask_beatsteps** np.ndarray of shape `(batch_size, max_beatsteps_seq_length)`
            - **attention_mask_extrapolated_beatstep** np.ndarray of shape `(batch_size,
              max_extrapolated_beatstep_seq_length)`
        """

        processed_features_dict = {}
        for feature_name, feature_value in inputs.items():
            if feature_name == "input_features":
                padded_feature_values, attention_mask = self._pad(feature_value, add_zero_line=True)
                processed_features_dict[feature_name] = padded_feature_values
                if return_attention_mask:
                    processed_features_dict["attention_mask"] = attention_mask
            else:
                padded_feature_values, attention_mask = self._pad(feature_value, add_zero_line=False)
                processed_features_dict[feature_name] = padded_feature_values
                if return_attention_mask:
                    processed_features_dict[f"attention_mask_{feature_name}"] = attention_mask

        # If we are processing only one example, we should remove the zero array line since we don't need it to
        # separate examples from each other.
        if not is_batched and not return_attention_mask:
            processed_features_dict["input_features"] = processed_features_dict["input_features"][:-1, ...]

        outputs = BatchFeature(processed_features_dict, tensor_type=return_tensors)

        return outputs

    def __call__(
        self,
        audio: Union[np.ndarray, list[float], list[np.ndarray], list[list[float]]],
        sampling_rate: Union[int, list[int]],
        steps_per_beat: int = 2,
        resample: Optional[bool] = True,
        return_attention_mask: Optional[bool] = False,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model.

        Args:
            audio (`np.ndarray`, `List`):
                The audio or batch of audio to be processed. Each audio can be a numpy array, a list of float values, a
                list of numpy arrays or a list of list of float values.
            sampling_rate (`int`):
                The sampling rate at which the `audio` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.
            steps_per_beat (`int`, *optional*, defaults to 2):
                This is used in interpolating `beat_times`.
            resample (`bool`, *optional*, defaults to `True`):
                Determines whether to resample the audio to `sampling_rate` or not before processing. Must be True
                during inference.
            return_attention_mask (`bool` *optional*, defaults to `False`):
                Denotes if attention_mask for input_features, beatsteps and extrapolated_beatstep will be given as
                output or not. Automatically set to True for batched inputs.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
                If nothing is specified, it will return list of `np.ndarray` arrays.
        """
        is_batched = bool(isinstance(audio, (list, tuple)) and isinstance(audio[0], (np.ndarray, tuple, list)))
        if is_batched:
            # This enables the user to process files of different sampling_rate at same time
            if not isinstance(sampling_rate, list):
                raise ValueError(
                    "Please give sampling_rate of each audio separately when you are passing multiple raw_audios at the same time. "
                    f"Received {sampling_rate}, expected [audio_1_sr, ..., audio_n_sr]."
                )
            return_attention_mask = True if return_attention_mask is None else return_attention_mask
        else:
            audio = [audio]
            sampling_rate = [sampling_rate]
            return_attention_mask = False if return_attention_mask is None else return_attention_mask

        batch_input_features, batch_beatsteps, batch_ext_beatstep = [], [], []
        total_len = len(audio)
        for index, (single_raw_audio, single_sampling_rate) in enumerate(zip(audio, sampling_rate)):
            bpm, beat_times, confidence, estimates, essentia_beat_intervals = self.extract_rhythm(
                audio=single_raw_audio
            )
            beatsteps = self.interpolate_beat_times(beat_times=beat_times, steps_per_beat=steps_per_beat, n_extend=1)
            if self.sampling_rate != single_sampling_rate and self.sampling_rate is not None:
                if resample:
                    # Change sampling_rate to self.sampling_rate
                    single_raw_audio = librosa_resample(
                        single_raw_audio,
                        orig_sr=single_sampling_rate,
                        target_sr=self.sampling_rate,
                        res_type="kaiser_best",
                    )
                else:
                    print(
                        f"The sampling_rate of the provided audio is different from the target sampling_rate "
                        f"of the Feature Extractor, {self.sampling_rate} vs {single_sampling_rate}. "
                        f"In these cases it is recommended to use `resample=True` in the `__call__` method to "
                        f"get the optimal behaviour."
                    )

            single_sampling_rate = self.sampling_rate
            start_sample = int(beatsteps[0] * single_sampling_rate)
            end_sample = int(beatsteps[-1] * single_sampling_rate)

            input_features, extrapolated_beatstep = self.preprocess_mel(
                single_raw_audio[start_sample:end_sample], beatsteps - beatsteps[0]
            )

            mel_specs = self.mel_spectrogram(input_features.astype(np.float32))

            # apply np.log to get log mel-spectrograms
            log_mel_specs = np.log(np.clip(mel_specs, a_min=1e-6, a_max=None))

            input_features = np.transpose(log_mel_specs, (0, -1, -2))

            batch_input_features.append(input_features)
            batch_beatsteps.append(beatsteps)
            batch_ext_beatstep.append(extrapolated_beatstep)
        output = BatchFeature(
            {
                "input_features": batch_input_features,
                "beatsteps": batch_beatsteps,
                "extrapolated_beatstep": batch_ext_beatstep,
            }
        )

        output = self.pad(
            output,
            is_batched=is_batched,
            return_attention_mask=return_attention_mask,
            return_tensors=return_tensors,
        )

        return output

VOCAB_FILES_NAMES = {
    "vocab": "vocab.json",
}

def token_time_to_note(number, cutoff_time_idx, current_idx):
    current_idx += number
    if cutoff_time_idx is not None:
        current_idx = min(current_idx, cutoff_time_idx)

    return current_idx

def token_note_to_note(number, current_velocity, default_velocity, note_onsets_ready, current_idx, notes):
    if note_onsets_ready[number] is not None:
        # offset with onset
        onset_idx = note_onsets_ready[number]
        if onset_idx < current_idx:
            # Time shift after previous note_on
            offset_idx = current_idx
            notes.append([onset_idx, offset_idx, number, default_velocity])
            onsets_ready = None if current_velocity == 0 else current_idx
            note_onsets_ready[number] = onsets_ready
    else:
        note_onsets_ready[number] = current_idx
    return notes

class Pop2PianoTokenizer(PreTrainedTokenizer):
    """
    Constructs a Pop2Piano tokenizer. This tokenizer does not require training.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab (`str`):
            Path to the vocab file which contains the vocabulary.
        default_velocity (`int`, *optional*, defaults to 77):
            Determines the default velocity to be used while creating midi Notes.
        num_bars (`int`, *optional*, defaults to 2):
            Determines cutoff_time_idx in for each token.
        unk_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"-1"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        eos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to 1):
            The end of sequence token.
        pad_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to 0):
             A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
            attention mechanisms or loss computation.
        bos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to 2):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
    """

    model_input_names = ["token_ids", "attention_mask"]
    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(
        self,
        vocab,
        default_velocity=77,
        num_bars=2,
        unk_token="-1",
        eos_token="1",
        pad_token="0",
        bos_token="2",
        **kwargs,
    ):
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token

        self.default_velocity = default_velocity
        self.num_bars = num_bars

        # Load the vocab
        with open(vocab, "rb") as file:
            self.encoder = json_load(file)

        # create mappings for encoder
        self.decoder = {v: k for k, v in self.encoder.items()}

        super().__init__(
            unk_token=unk_token,
            eos_token=eos_token,
            pad_token=pad_token,
            bos_token=bos_token,
            **kwargs,
        )

    @property
    def vocab_size(self):
        """Returns the vocabulary size of the tokenizer."""
        return len(self.encoder)

    def get_vocab(self):
        """Returns the vocabulary of the tokenizer."""
        return dict(self.encoder, **self.added_tokens_encoder)

    def _convert_id_to_token(self, token_id: int) -> list:
        """
        Decodes the token ids generated by the transformer into notes.

        Args:
            token_id (`int`):
                This denotes the ids generated by the transformers to be converted to Midi tokens.

        Returns:
            `List`: A list consists of token_type (`str`) and value (`int`).
        """

        token_type_value = self.decoder.get(token_id, f"{self.unk_token}_TOKEN_TIME")
        token_type_value = token_type_value.split("_")
        token_type, value = "_".join(token_type_value[1:]), int(token_type_value[0])

        return [token_type, value]

    def _convert_token_to_id(self, token, token_type="TOKEN_TIME") -> int:
        """
        Encodes the Midi tokens to transformer generated token ids.

        Args:
            token (`int`):
                This denotes the token value.
            token_type (`str`):
                This denotes the type of the token. There are four types of midi tokens such as "TOKEN_TIME",
                "TOKEN_VELOCITY", "TOKEN_NOTE" and "TOKEN_SPECIAL".

        Returns:
            `int`: returns the id of the token.
        """
        return self.encoder.get(f"{token}_{token_type}", int(self.unk_token))

    def relative_batch_tokens_ids_to_notes(
        self,
        tokens: np.ndarray,
        beat_offset_idx: int,
        bars_per_batch: int,
        cutoff_time_idx: int,
    ):
        """
        Converts relative tokens to notes which are then used to generate pretty midi object.

        Args:
            tokens (`np.ndarray`):
                Tokens to be converted to notes.
            beat_offset_idx (`int`):
                Denotes beat offset index for each note in generated Midi.
            bars_per_batch (`int`):
                A parameter to control the Midi output generation.
            cutoff_time_idx (`int`):
                Denotes the cutoff time index for each note in generated Midi.
        """

        notes = None

        for index in range(len(tokens)):
            _tokens = tokens[index]
            _start_idx = beat_offset_idx + index * bars_per_batch * 4
            _cutoff_time_idx = cutoff_time_idx + _start_idx
            _notes = self.relative_tokens_ids_to_notes(
                _tokens,
                start_idx=_start_idx,
                cutoff_time_idx=_cutoff_time_idx,
            )

            if len(_notes) == 0:
                pass
            elif notes is None:
                notes = _notes
            else:
                notes = np.concatenate((notes, _notes), axis=0)

        if notes is None:
            return []
        return notes

    def relative_batch_tokens_ids_to_midi(
        self,
        tokens: np.ndarray,
        beatstep: np.ndarray,
        beat_offset_idx: int = 0,
        bars_per_batch: int = 2,
        cutoff_time_idx: int = 12,
    ):
        """
        Converts tokens to Midi. This method calls `relative_batch_tokens_ids_to_notes` method to convert batch tokens
        to notes then uses `notes_to_midi` method to convert them to Midi.

        Args:
            tokens (`np.ndarray`):
                Denotes tokens which alongside beatstep will be converted to Midi.
            beatstep (`np.ndarray`):
                We get beatstep from feature extractor which is also used to get Midi.
            beat_offset_idx (`int`, *optional*, defaults to 0):
                Denotes beat offset index for each note in generated Midi.
            bars_per_batch (`int`, *optional*, defaults to 2):
                A parameter to control the Midi output generation.
            cutoff_time_idx (`int`, *optional*, defaults to 12):
                Denotes the cutoff time index for each note in generated Midi.
        """
        beat_offset_idx = 0 if beat_offset_idx is None else beat_offset_idx
        notes = self.relative_batch_tokens_ids_to_notes(
            tokens=tokens,
            beat_offset_idx=beat_offset_idx,
            bars_per_batch=bars_per_batch,
            cutoff_time_idx=cutoff_time_idx,
        )
        midi = self.notes_to_midi(notes, beatstep, offset_sec=beatstep[beat_offset_idx])
        return midi

    # Taken from the original code
    # Please see https://github.com/sweetcocoa/pop2piano/blob/fac11e8dcfc73487513f4588e8d0c22a22f2fdc5/midi_tokenizer.py#L257
    def relative_tokens_ids_to_notes(
        self, tokens: np.ndarray, start_idx: float, cutoff_time_idx: Optional[float] = None
    ):
        """
        Converts relative tokens to notes which will then be used to create Pretty Midi objects.

        Args:
            tokens (`np.ndarray`):
                Relative Tokens which will be converted to notes.
            start_idx (`float`):
                A parameter which denotes the starting index.
            cutoff_time_idx (`float`, *optional*):
                A parameter used while converting tokens to notes.
        """
        words = [self._convert_id_to_token(token) for token in tokens]

        current_idx = start_idx
        current_velocity = 0
        note_onsets_ready = [None for i in range(sum([k.endswith("NOTE") for k in self.encoder.keys()]) + 1)]
        notes = []
        for token_type, number in words:
            if token_type == "TOKEN_SPECIAL":
                if number == 1:
                    break
            elif token_type == "TOKEN_TIME":
                current_idx = token_time_to_note(
                    number=number, cutoff_time_idx=cutoff_time_idx, current_idx=current_idx
                )
            elif token_type == "TOKEN_VELOCITY":
                current_velocity = number

            elif token_type == "TOKEN_NOTE":
                notes = token_note_to_note(
                    number=number,
                    current_velocity=current_velocity,
                    default_velocity=self.default_velocity,
                    note_onsets_ready=note_onsets_ready,
                    current_idx=current_idx,
                    notes=notes,
                )
            else:
                raise ValueError("Token type not understood!")

        for pitch, note_onset in enumerate(note_onsets_ready):
            # force offset if no offset for each pitch
            if note_onset is not None:
                if cutoff_time_idx is None:
                    cutoff = note_onset + 1
                else:
                    cutoff = max(cutoff_time_idx, note_onset + 1)

                offset_idx = max(current_idx, cutoff)
                notes.append([note_onset, offset_idx, pitch, self.default_velocity])

        if len(notes) == 0:
            return []
        else:
            notes = np.array(notes)
            note_order = notes[:, 0] * 128 + notes[:, 1]
            notes = notes[note_order.argsort()]
            return notes

    def notes_to_midi(self, notes: np.ndarray, beatstep: np.ndarray, offset_sec: int = 0.0):
        """
        Converts notes to Midi.

        Args:
            notes (`np.ndarray`):
                This is used to create Pretty Midi objects.
            beatstep (`np.ndarray`):
                This is the extrapolated beatstep that we get from feature extractor.
            offset_sec (`int`, *optional*, defaults to 0.0):
                This represents the offset seconds which is used while creating each Pretty Midi Note.
        """
        new_pm = pretty_midi_fix.PrettyMIDI(resolution=384, initial_tempo=120.0)
        new_inst = pretty_midi_fix.Instrument(program=0)
        new_notes = []

        for onset_idx, offset_idx, pitch, velocity in notes:
            new_note = pretty_midi_fix.Note(
                velocity=velocity,
                pitch=pitch,
                start=beatstep[onset_idx] - offset_sec,
                end=beatstep[offset_idx] - offset_sec,
            )
            new_notes.append(new_note)
        new_inst.notes = new_notes
        new_pm.instruments.append(new_inst)
        new_pm.remove_invalid_notes()
        return new_pm

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple[str]:
        """
        Saves the tokenizer's vocabulary dictionary to the provided save_directory.

        Args:
            save_directory (`str`):
                A path to the directory where to saved. It will be created if it doesn't exist.
            filename_prefix (`Optional[str]`, *optional*):
                A prefix to add to the names of the files saved by the tokenizer.
        """
        if not os.path.isdir(save_directory):
            print(f"Vocabulary path ({save_directory}) should be a directory")
            return

        # Save the encoder.
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab"]
        )
        with open(out_vocab_file, "w") as file:
            file.write(json_dumps(self.encoder))

        return (out_vocab_file,)

    def encode_plus(
        self,
        notes: Union[np.ndarray, list[pretty_midi_fix.Note]],
        truncation_strategy: Optional[TruncationStrategy] = None,
        max_length: Optional[int] = None,
        **kwargs,
    ) -> BatchEncoding:
        r"""
        This is the `encode_plus` method for `Pop2PianoTokenizer`. It converts the midi notes to the transformer
        generated token ids. It only works on a single batch, to process multiple batches please use
        `batch_encode_plus` or `__call__` method.

        Args:
            notes (`np.ndarray` of shape `[sequence_length, 4]` or `list` of `pretty_midi_fix.Note` objects):
                This represents the midi notes. If `notes` is a `np.ndarray`:
                    - Each sequence must have 4 values, they are `onset idx`, `offset idx`, `pitch` and `velocity`.
                If `notes` is a `list` containing `pretty_midi_fix.Note` objects:
                    - Each sequence must have 4 attributes, they are `start`, `end`, `pitch` and `velocity`.
            truncation_strategy ([`~tokenization_utils_base.TruncationStrategy`], *optional*):
                Indicates the truncation strategy that is going to be used during truncation.
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).

        Returns:
            `BatchEncoding` containing the tokens ids.
        """
        # check if notes is a pretty_midi_fix object or not, if yes then extract the attributes and put them into a numpy
        # array.
        if isinstance(notes[0], pretty_midi_fix.Note):
            notes = np.array(
                [[each_note.start, each_note.end, each_note.pitch, each_note.velocity] for each_note in notes]
            ).reshape(-1, 4)

        # to round up all the values to the closest int values.
        notes = np.round(notes).astype(np.int32)
        max_time_idx = notes[:, :2].max()

        times = [[] for i in range(max_time_idx + 1)]
        for onset, offset, pitch, velocity in notes:
            times[onset].append([pitch, velocity])
            times[offset].append([pitch, 0])

        tokens = []
        current_velocity = 0
        for i, time in enumerate(times):
            if len(time) == 0:
                continue
            tokens.append(self._convert_token_to_id(i, "TOKEN_TIME"))
            for pitch, velocity in time:
                velocity = int(velocity > 0)
                if current_velocity != velocity:
                    current_velocity = velocity
                    tokens.append(self._convert_token_to_id(velocity, "TOKEN_VELOCITY"))
                tokens.append(self._convert_token_to_id(pitch, "TOKEN_NOTE"))

        total_len = len(tokens)

        # truncation
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and total_len > max_length:
            tokens, _, _ = self.truncate_sequences(
                ids=tokens,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation_strategy,
                **kwargs,
            )

        return BatchEncoding({"token_ids": tokens})

    def batch_encode_plus(
        self,
        notes: Union[np.ndarray, list[pretty_midi_fix.Note]],
        truncation_strategy: Optional[TruncationStrategy] = None,
        max_length: Optional[int] = None,
        **kwargs,
    ) -> BatchEncoding:
        r"""
        This is the `batch_encode_plus` method for `Pop2PianoTokenizer`. It converts the midi notes to the transformer
        generated token ids. It works on multiple batches by calling `encode_plus` multiple times in a loop.

        Args:
            notes (`np.ndarray` of shape `[batch_size, sequence_length, 4]` or `list` of `pretty_midi_fix.Note` objects):
                This represents the midi notes. If `notes` is a `np.ndarray`:
                    - Each sequence must have 4 values, they are `onset idx`, `offset idx`, `pitch` and `velocity`.
                If `notes` is a `list` containing `pretty_midi_fix.Note` objects:
                    - Each sequence must have 4 attributes, they are `start`, `end`, `pitch` and `velocity`.
            truncation_strategy ([`~tokenization_utils_base.TruncationStrategy`], *optional*):
                Indicates the truncation strategy that is going to be used during truncation.
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).

        Returns:
            `BatchEncoding` containing the tokens ids.
        """

        encoded_batch_token_ids = []
        for i in range(len(notes)):
            encoded_batch_token_ids.append(
                self.encode_plus(
                    notes[i],
                    truncation_strategy=truncation_strategy,
                    max_length=max_length,
                    **kwargs,
                )["token_ids"]
            )

        return BatchEncoding({"token_ids": encoded_batch_token_ids})

    def __call__(
        self,
        notes: Union[
            np.ndarray,
            list[pretty_midi_fix.Note],
            list[list[pretty_midi_fix.Note]],
        ],
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        r"""
        This is the `__call__` method for `Pop2PianoTokenizer`. It converts the midi notes to the transformer generated
        token ids.

        Args:
            notes (`np.ndarray` of shape `[batch_size, max_sequence_length, 4]` or `list` of `pretty_midi_fix.Note` objects):
                This represents the midi notes.

                If `notes` is a `np.ndarray`:
                    - Each sequence must have 4 values, they are `onset idx`, `offset idx`, `pitch` and `velocity`.
                If `notes` is a `list` containing `pretty_midi_fix.Note` objects:
                    - Each sequence must have 4 attributes, they are `start`, `end`, `pitch` and `velocity`.
            padding (`bool`, `str` or [`~file_utils.PaddingStrategy`], *optional*, defaults to `False`):
                Activates and controls padding. Accepts the following values:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            truncation (`bool`, `str` or [`~tokenization_utils_base.TruncationStrategy`], *optional*, defaults to `False`):
                Activates and controls truncation. Accepts the following values:

                - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or
                  to the maximum acceptable input length for the model if that argument is not provided. This will
                  truncate token by token, removing a token from the longest sequence in the pair if a pair of
                  sequences (or a batch of pairs) is provided.
                - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
                  greater than the model maximum admissible input size).
            max_length (`int`, *optional*):
                Controls the maximum length to use by one of the truncation/padding parameters. If left unset or set to
                `None`, this will use the predefined model maximum length if a maximum length is required by one of the
                truncation/padding parameters. If the model has no specific maximum input length (like XLNet)
                truncation/padding to a maximum length will be deactivated.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
                the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific tokenizer's default, defined by the `return_outputs` attribute.

                [What are attention masks?](../glossary#attention-mask)
            return_tensors (`str` or [`~file_utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            verbose (`bool`, *optional*, defaults to `True`):
                Whether or not to print more information and warnings.

        Returns:
            `BatchEncoding` containing the token_ids.
        """

        # check if it is batched or not
        # it is batched if its a list containing a list of `pretty_midi_fix.Notes` where the outer list contains all the
        # batches and the inner list contains all Notes for a single batch. Otherwise if np.ndarray is passed it will be
        # considered batched if it has shape of `[batch_size, seqence_length, 4]` or ndim=3.
        is_batched = notes.ndim == 3 if isinstance(notes, np.ndarray) else isinstance(notes[0], list)

        # get the truncation and padding strategy
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        if is_batched:
            # If the user has not explicitly mentioned `return_attention_mask` as False, we change it to True
            return_attention_mask = True if return_attention_mask is None else return_attention_mask
            token_ids = self.batch_encode_plus(
                notes=notes,
                truncation_strategy=truncation_strategy,
                max_length=max_length,
                **kwargs,
            )
        else:
            token_ids = self.encode_plus(
                notes=notes,
                truncation_strategy=truncation_strategy,
                max_length=max_length,
                **kwargs,
            )

        # since we already have truncated sequnences we are just left to do padding
        token_ids = self.pad(
            token_ids,
            padding=padding_strategy,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_tensors=return_tensors,
            verbose=verbose,
        )

        return token_ids

    def batch_decode(
        self,
        token_ids,
        feature_extractor_output: BatchFeature,
        return_midi: bool = True,
    ):
        r"""
        This is the `batch_decode` method for `Pop2PianoTokenizer`. It converts the token_ids generated by the
        transformer to midi_notes and returns them.

        Args:
            token_ids (`Union[np.ndarray, torch.Tensor, tf.Tensor]`):
                Output token_ids of `Pop2PianoConditionalGeneration` model.
            feature_extractor_output (`BatchFeature`):
                Denotes the output of `Pop2PianoFeatureExtractor.__call__`. It must contain `"beatstep"` and
                `"extrapolated_beatstep"`. Also `"attention_mask_beatsteps"` and
                `"attention_mask_extrapolated_beatstep"`
                 should be present if they were returned by the feature extractor.
            return_midi (`bool`, *optional*, defaults to `True`):
                Whether to return midi object or not.
        Returns:
            If `return_midi` is True:
                - `BatchEncoding` containing both `notes` and `pretty_midi_fix.pretty_midi_fix.PrettyMIDI` objects.
            If `return_midi` is False:
                - `BatchEncoding` containing `notes`.
        """

        # check if they have attention_masks(attention_mask, attention_mask_beatsteps, attention_mask_extrapolated_beatstep) or not
        attention_masks_present = bool(
            hasattr(feature_extractor_output, "attention_mask")
            and hasattr(feature_extractor_output, "attention_mask_beatsteps")
            and hasattr(feature_extractor_output, "attention_mask_extrapolated_beatstep")
        )

        # if we are processing batched inputs then we must need attention_masks
        if not attention_masks_present and feature_extractor_output["beatsteps"].shape[0] > 1:
            raise ValueError(
                "attention_mask, attention_mask_beatsteps and attention_mask_extrapolated_beatstep must be present "
                "for batched inputs! But one of them were not present."
            )

        # check for length mismatch between inputs_embeds, beatsteps and extrapolated_beatstep
        if attention_masks_present:
            # since we know about the number of examples in token_ids from attention_mask
            if (
                sum(feature_extractor_output["attention_mask"][:, 0] == 0)
                != feature_extractor_output["beatsteps"].shape[0]
                or feature_extractor_output["beatsteps"].shape[0]
                != feature_extractor_output["extrapolated_beatstep"].shape[0]
            ):
                raise ValueError(
                    "Length mistamtch between token_ids, beatsteps and extrapolated_beatstep! Found "
                    f"token_ids length - {token_ids.shape[0]}, beatsteps shape - {feature_extractor_output['beatsteps'].shape[0]} "
                    f"and extrapolated_beatsteps shape - {feature_extractor_output['extrapolated_beatstep'].shape[0]}"
                )
            if feature_extractor_output["attention_mask"].shape[0] != token_ids.shape[0]:
                raise ValueError(
                    f"Found attention_mask of length - {feature_extractor_output['attention_mask'].shape[0]} but token_ids of length - {token_ids.shape[0]}"
                )
        else:
            # if there is no attention mask present then it's surely a single example
            if (
                feature_extractor_output["beatsteps"].shape[0] != 1
                or feature_extractor_output["extrapolated_beatstep"].shape[0] != 1
            ):
                raise ValueError(
                    "Length mistamtch of beatsteps and extrapolated_beatstep! Since attention_mask is not present the number of examples must be 1, "
                    f"But found beatsteps length - {feature_extractor_output['beatsteps'].shape[0]}, extrapolated_beatsteps length - {feature_extractor_output['extrapolated_beatstep'].shape[0]}."
                )

        if attention_masks_present:
            # check for zeros(since token_ids are separated by zero arrays)
            batch_idx = np.where(feature_extractor_output["attention_mask"][:, 0] == 0)[0]
        else:
            batch_idx = [token_ids.shape[0]]

        notes_list = []
        pretty_midi_fix_objects_list = []
        start_idx = 0
        for index, end_idx in enumerate(batch_idx):
            each_tokens_ids = token_ids[start_idx:end_idx]
            # check where the whole example ended by searching for eos_token_id and getting the upper bound
            each_tokens_ids = each_tokens_ids[:, : np.max(np.where(each_tokens_ids == int(self.eos_token))[1]) + 1]
            beatsteps = feature_extractor_output["beatsteps"][index]
            extrapolated_beatstep = feature_extractor_output["extrapolated_beatstep"][index]

            # if attention mask is present then mask out real array/tensor
            if attention_masks_present:
                attention_mask_beatsteps = feature_extractor_output["attention_mask_beatsteps"][index]
                attention_mask_extrapolated_beatstep = feature_extractor_output[
                    "attention_mask_extrapolated_beatstep"
                ][index]
                beatsteps = beatsteps[: np.max(np.where(attention_mask_beatsteps == 1)[0]) + 1]
                extrapolated_beatstep = extrapolated_beatstep[
                    : np.max(np.where(attention_mask_extrapolated_beatstep == 1)[0]) + 1
                ]

            each_tokens_ids = to_numpy(each_tokens_ids)
            beatsteps = to_numpy(beatsteps)
            extrapolated_beatstep = to_numpy(extrapolated_beatstep)

            pretty_midi_fix_object = self.relative_batch_tokens_ids_to_midi(
                tokens=each_tokens_ids,
                beatstep=extrapolated_beatstep,
                bars_per_batch=self.num_bars,
                cutoff_time_idx=(self.num_bars + 1) * 4,
            )

            for note in pretty_midi_fix_object.instruments[0].notes:
                note.start += beatsteps[0]
                note.end += beatsteps[0]
                notes_list.append(note)

            pretty_midi_fix_objects_list.append(pretty_midi_fix_object)
            start_idx += end_idx + 1  # 1 represents the zero array

        if return_midi:
            return BatchEncoding({"notes": notes_list, "pretty_midi_objects": pretty_midi_fix_objects_list})

        return BatchEncoding({"notes": notes_list})

class Pop2PianoProcessor(ProcessorMixin):
    r"""
    Constructs an Pop2Piano processor which wraps a Pop2Piano Feature Extractor and Pop2Piano Tokenizer into a single
    processor.

    [`Pop2PianoProcessor`] offers all the functionalities of [`Pop2PianoFeatureExtractor`] and [`Pop2PianoTokenizer`].
    See the docstring of [`~Pop2PianoProcessor.__call__`] and [`~Pop2PianoProcessor.decode`] for more information.

    Args:
        feature_extractor (`Pop2PianoFeatureExtractor`):
            An instance of [`Pop2PianoFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`Pop2PianoTokenizer`):
            An instance of ['Pop2PianoTokenizer`]. The tokenizer is a required input.
    """

    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "Pop2PianoFeatureExtractor"
    tokenizer_class = "Pop2PianoTokenizer"

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    def __call__(
        self,
        audio: Union[np.ndarray, list[float], list[np.ndarray]] = None,
        sampling_rate: Optional[Union[int, list[int]]] = None,
        steps_per_beat: int = 2,
        resample: Optional[bool] = True,
        notes: Union[list, TensorType] = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        verbose: bool = True,
        **kwargs,
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        This method uses [`Pop2PianoFeatureExtractor.__call__`] method to prepare log-mel-spectrograms for the model,
        and [`Pop2PianoTokenizer.__call__`] to prepare token_ids from notes.

        Please refer to the docstring of the above two methods for more information.
        """

        # Since Feature Extractor needs both audio and sampling_rate and tokenizer needs both token_ids and
        # feature_extractor_output, we must check for both.
        if (audio is None and sampling_rate is None) and (notes is None):
            raise ValueError(
                "You have to specify at least audios and sampling_rate in order to use feature extractor or "
                "notes to use the tokenizer part."
            )

        if audio is not None and sampling_rate is not None:
            inputs = self.feature_extractor(
                audio=audio,
                sampling_rate=sampling_rate,
                steps_per_beat=steps_per_beat,
                resample=resample,
                **kwargs,
            )

        if notes is not None:
            encoded_token_ids = self.tokenizer(
                notes=notes,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                pad_to_multiple_of=pad_to_multiple_of,
                verbose=verbose,
                **kwargs,
            )

        if notes is None:
            return inputs

        elif audio is None or sampling_rate is None:
            return encoded_token_ids

        else:
            inputs["token_ids"] = encoded_token_ids["token_ids"]
            return inputs

    def batch_decode(
        self,
        token_ids,
        feature_extractor_output: BatchFeature,
        return_midi: bool = True,
    ) -> BatchEncoding:
        """
        This method uses [`Pop2PianoTokenizer.batch_decode`] method to convert model generated token_ids to midi_notes.

        Please refer to the docstring of the above two methods for more information.
        """

        return self.tokenizer.batch_decode(
            token_ids=token_ids, feature_extractor_output=feature_extractor_output, return_midi=return_midi
        )

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + feature_extractor_input_names))

    def save_pretrained(self, save_directory, **kwargs):
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)
        return super().save_pretrained(save_directory, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        args = cls._get_arguments_from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(*args)



class Pop2Piano:
    def __init__(self,device="cpu",model_path=snapshot_download("sweetcocoa/pop2piano")):
        self.model = Pop2PianoForConditionalGeneration.from_pretrained(model_path).to(device)
        self.processor = Pop2PianoProcessor.from_pretrained(model_path)
    
    def predict(self,audio,composer=1,num_bars=2,num_beams=1,steps_per_beat=2,output_file="output.mid"):
        data, sr = librosa_load(audio, sr=None)
        inputs = self.processor(data, sr, steps_per_beat,return_tensors="pt",num_bars=num_bars)
        self.processor.batch_decode(self.model.generate(num_beams=num_beams,do_sample=True,input_features=inputs["input_features"], composer="composer" + str(composer)),inputs)["pretty_midi_objects"][0].write(open(output_file, "wb"))
        return output_file
