#integrating the neccessary classes
import torch
import torch.nn as nn
import math
from positional_encodings import PositionalEncoding
from multihead_attention import MultiHeadAttention
from encoding_layers import position_wide_feed_forward, Residual_layer
from masking_for_attention import mask

class EncoderLayer(nn.Module):
    def __init__(self, dimension_for_model, num_of_heads, dim_feedforward, dropout = 0.1):
        '''
        dimension_for_model: the dimension desired for the model specified at the embeddings layer
        num_of_heads: the number of heads for the multi-head-attention structure to keep track of
        dim_feedforward: the dimension of the positional feed forward structure
        dropout: structure for removing model dependencies during training, improving robustness
        '''
        super().__init__()
        #loading previously coded structures for multi-head attention
        self.self_attn = MultiHeadAttention(dimension_for_model, num_of_heads, dropout)
        self.norm1 = nn.LayerNorm(dimension_for_model)
        self.dropout1 = nn.Dropout(dropout)
        #loading previously coded structures for position_wide_feed_forward
        self.ffn = position_wide_feed_forward(dimension_for_model, dim_feedforward, dropout)
        self.norm2 = nn.LayerNorm(dimension_for_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention block
        _src = src
        attn_output, _ = self.self_attn(src, src, src, mask=src_mask)
        src = self.norm1(_src + self.dropout1(attn_output))
        # Feed-forward block
        _src = src
        ff_output = self.ffn(src)
        src = self.norm2(_src + self.dropout2(ff_output))
        return src


class Encoder(nn.Module):
    """
    Stacked Transformer encoder:
      - embedding + positional encoding
      - N encoder layers
      - final layer norm
    """
    def __init__(self, vocab_size, dimension_of_model, num_of_heads, num_layers, dim_feedforward = 2048, dropout = 0.1, max_len = 5000):
        super().__init__()
        # token embeddings
        self.embed = nn.Embedding(vocab_size, dimension_of_model)
        # positional encodings (sinusoidal or learned)
        self.pe = PositionalEncoding(dimension_of_model, dropout, max_len)
        # stacked encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(dimension_of_model, num_of_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        # final normalization
        self.norm = nn.LayerNorm(dimension_of_model)

    def forward(self, src_ids, src_mask = None) -> torch.Tensor:
        """
        Args:
          src_ids: [batch_size x seq_len] input token indices
          src_mask: [seq_len x seq_len] mask to prevent attending to future tokens
        """
        # embed tokens and scale
        x = self.embed(src_ids) * math.sqrt(self.embed.embedding_dim)
        # add positional information
        x = self.pe(x)
        # pass through each encoder layer
        for layer in self.layers:
            x = layer(x, src_mask)
        # final layer normalization
        return self.norm(x)
    
