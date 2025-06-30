#integrating the neccessary classes
import torch
import torch.nn as nn
import math
from .positional_encodings import PositionalEncoding  #import other modules neccessary for 
from .multihead_attention import MultiHeadAttention
from .encoding_layers import position_wide_feed_forward, Residual_layer
from .masking_for_attention import mask
from .embeddings import Embeddings

class EncoderLayer(nn.Module):
    def __init__(self, dimension_for_model, num_of_heads, dim_feedforward, dropout = 0.1):
        '''
        dimension_for_model: the dimension desired for the model specified at the embeddings layer
        num_of_heads: the number of heads for the multi-head-attention structure to keep track of
        dim_feedforward: the dimension of the positional feed forward structure
        dropout: structure for removing model dependencies during training, improving robustness
        '''
        super().__init__()
        # Loading previously coded structures for multi-head attention
        self.self_attn = MultiHeadAttention(dimension_for_model, num_of_heads, dropout)
        self.norm1 = nn.LayerNorm(dimension_for_model)
        self.dropout1 = nn.Dropout(dropout)
        # Loading previously coded structures for position_wide_feed_forward
        self.ffn = position_wide_feed_forward(dimension_for_model, dim_feedforward, dropout)
        self.norm2 = nn.LayerNorm(dimension_for_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention block
        _src = src
        attn_output, _ = self.self_attn(src, src, src, mask=src_mask)  
        src = self.norm1(_src + self.dropout1(attn_output))  # changed attention output
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
    def __init__(self, vocab_size, dimension_of_model, num_of_heads, num_layers, dim_feedforward = 2048, dropout = 0.1, max_len = 5000, num_of_roles=2, max_turns=16):
        super().__init__()
        # Token/role/turn embeddings
        self.embed = Embeddings(vocab_size, dimension_for_model=dimension_of_model, num_of_roles=num_of_roles, max_turns=max_turns)
        # Positional encodings (sinusoidal or learned)
        self.pe = PositionalEncoding(dimension_of_model, dropout=dropout, max_len=max_len)
        # Stacked encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(dimension_of_model, num_of_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        # Final normalization
        self.norm = nn.LayerNorm(dimension_of_model)

    def forward(self, src_ids, roles, turns, src_mask = None) -> torch.Tensor:
        """
        Args:
          src_ids: [batch_size x seq_len] input token indices
          roles:   [batch_size x seq_len] role ids
          turns:   [batch_size x seq_len] turn ids
          src_mask: [batch_size, 1, 1, seq_len] mask to prevent attending to padding tokens
        """
        # Embed tokens, roles, and turns
        x = self.embed(src_ids, roles, turns)
        # Add positional information
        x = self.pe(x)
        # Pass through each encoder layer
        for layer in self.layers:
            x = layer(x, src_mask)
        # Final layer normalization
        return self.norm(x)
    
    def load_state_dict(self, state_dict, strict=True):
        """
        Custom state dict loading to handle backward compatibility with old model format
        """
        # Check if this is an old model format (has encoder.embed.weight)
        if 'encoder.embed.weight' in state_dict:
            # This is an old model, we need to adapt the weights
            old_embed_weight = state_dict['encoder.embed.weight']
            
            # Copy the old embedding weights to the new structure
            state_dict['encoder.embed.lut.weight'] = old_embed_weight
            state_dict['encoder.embed.lut_roles.weight'] = torch.zeros_like(old_embed_weight)
            state_dict['encoder.embed.lut_turns.weight'] = torch.zeros_like(old_embed_weight)
            state_dict['encoder.embed.norm.weight'] = torch.ones(old_embed_weight.size(1))
            state_dict['encoder.embed.norm.bias'] = torch.zeros(old_embed_weight.size(1))
            
            # Remove the old key
            del state_dict['encoder.embed.weight']
        
        return super().load_state_dict(state_dict, strict=strict)
    
