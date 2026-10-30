"""Numerical CGM building blocks; legacy v2 implementations are archived outside X-Life-2."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from configuration import configured, settings

def causal_mask(length, device, memory_length=None):
    columns = length if memory_length is None else memory_length
    return torch.arange(columns, device=device).unsqueeze(0) > torch.arange(length, device=device).unsqueeze(1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=None, time_encoding_base=None):
        super().__init__()
        max_len = configured('model', 'position_max_length', max_len)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(configured('model', 'time_encoding_base', time_encoding_base)) / d_model))
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * frequencies)
        encoding[:, 1::2] = torch.cos(position * frequencies[:encoding[:, 1::2].size(1)])
        self.register_buffer("pe", encoding.unsqueeze(1))

    def forward(self, sequence):
        if sequence.size(0) > self.pe.size(0):
            raise ValueError("Sequence exceeds the configured positional encoding length.")
        return sequence + self.pe[:sequence.size(0)].to(dtype=sequence.dtype)


def transformer_encoder(dimension, heads, layers, dropout, feedforward=None, activation=None):
    if dimension % heads:
        raise ValueError("Transformer state dimension must be divisible by its number of heads.")
    layer = nn.TransformerEncoderLayer(d_model=dimension, nhead=heads,
                                      dim_feedforward=dimension * settings('model', 'feedforward_multiplier') if feedforward is None else feedforward,
                                      dropout=dropout, activation=configured('model', 'activation', activation), batch_first=True)
    return nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)


class CGMTransformerEncoder(nn.Module):
    """Non-overlapping CGM patches with explicit within-patch validity bits."""
    def __init__(self, input_dim=1, d_model=None, nhead=None, num_layers=None,
                 dim_feedforward=None, dropout=None, patch_size=None, position_max_length=None, activation=None, time_encoding_base=None):
        super().__init__()
        d_model = configured('model', 'dimension', d_model)
        nhead = configured('model', 'heads', nhead)
        num_layers = configured('model', 'layers', num_layers)
        dim_feedforward = configured('model', 'feedforward', dim_feedforward)
        dropout = configured('model', 'dropout', dropout)
        patch_size = configured('model', 'patch_size', patch_size)
        if input_dim != 1 or patch_size < 1:
            raise ValueError("CGM is scalar-valued and patch_size must be positive.")
        self.d_model, self.patch_size = d_model, int(patch_size)
        self.input_projection = nn.Linear(self.patch_size * 2, d_model)
        self.pos_encoder = PositionalEncoding(d_model, position_max_length, time_encoding_base)
        self.transformer_encoder = transformer_encoder(d_model, nhead, num_layers, dropout, dim_feedforward, activation)

    def patchify(self, cgm_sequence, attention_mask=None):
        if cgm_sequence.ndim != 2 or cgm_sequence.size(1) < 1:
            raise ValueError("CGM must have shape [batch, nonempty time].")
        valid = torch.isfinite(cgm_sequence)
        if attention_mask is not None:
            if attention_mask.shape != cgm_sequence.shape:
                raise ValueError("CGM values and masks must have the same shape.")
            valid = valid & attention_mask.to(device=cgm_sequence.device, dtype=torch.bool)
        values = torch.where(valid, cgm_sequence, torch.zeros_like(cgm_sequence))
        padding = (-values.size(1)) % self.patch_size
        values = F.pad(values, (0, padding))
        valid = F.pad(valid, (0, padding), value=False)
        values = values.reshape(values.size(0), -1, self.patch_size)
        valid = valid.reshape(valid.size(0), -1, self.patch_size)
        return values, valid

    def patch_weights(self, cgm_sequence, attention_mask=None):
        return self.patchify(cgm_sequence, attention_mask)[1].sum(dim=-1)

    def forward(self, cgm_sequence, attention_mask=None):
        values, valid = self.patchify(cgm_sequence, attention_mask)
        tokens = self.input_projection(torch.cat([values, valid.to(values.dtype)], dim=-1))
        tokens = self.pos_encoder(tokens.transpose(0, 1)).transpose(0, 1)
        padding = ~valid.any(dim=-1)


        empty = padding.all(dim=1)
        padding = padding.clone()
        padding[empty, 0] = False
        return self.transformer_encoder(tokens, src_key_padding_mask=padding)


class CGMTransformerDecoder(nn.Module):
    def __init__(self, d_model=None, output_dim=1, nhead=None, num_layers=None,
                 dim_feedforward=None, dropout=None, position_max_length=None, activation=None, time_encoding_base=None):
        super().__init__()
        d_model = configured('model', 'dimension', d_model)
        nhead = configured('model', 'heads', nhead)
        num_layers = configured('model', 'layers', num_layers)
        dim_feedforward = configured('model', 'feedforward', dim_feedforward)
        dropout = configured('model', 'dropout', dropout)
        self.d_model = d_model
        self.target_input_projection = nn.Linear(output_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, position_max_length, time_encoding_base)
        layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                           activation=configured('model', 'activation', activation), batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(layer, num_layers)
        self.output_projection = nn.Linear(d_model, output_dim)

    def _decode(self, decoder_inputs, memory):
        length = decoder_inputs.size(1)
        embedded = self.target_input_projection(decoder_inputs)
        embedded = self.pos_encoder(embedded.transpose(0, 1)).transpose(0, 1)
        decoded = self.transformer_decoder(
            embedded, memory, tgt_mask=causal_mask(length, embedded.device),
            memory_mask=causal_mask(length, embedded.device, memory.size(1))
        )
        return self.output_projection(decoded)

    def forward(self, encoded_cgm, encoded_text, target_sequence=None, target_len=None):
        memory = encoded_text.unsqueeze(1) if encoded_text.ndim == 2 else encoded_text
        if memory.ndim != 3:
            raise ValueError("Decoder memory must have shape [batch, horizon, state].")
        if target_sequence is not None:
            inputs = torch.zeros_like(target_sequence)
            inputs[:, 1:] = target_sequence[:, :-1]
            return self._decode(inputs, memory)
        horizon = memory.size(1) if target_len is None else int(target_len)
        if horizon < 1:
            raise ValueError("Prediction horizon must be positive.")
        start = memory.new_zeros(memory.size(0), 1, 1)
        history = start
        generated = []
        for step in range(horizon):
            next_value = self._decode(history, memory[:, :step + 1])[:, -1:]
            generated.append(next_value)
            history = torch.cat([history, next_value], dim=1)
        return torch.cat(generated, dim=1)


