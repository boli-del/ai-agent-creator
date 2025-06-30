import torch
import torch.nn as nn
import math
from .positional_encodings import PositionalEncoding
from .encoding_layers import position_wide_feed_forward

class DecoderLayer(nn.Module):
    def __init__(self, dimension_for_model, num_of_heads, dim_feedforward=2048, dropout=0.1):

        '''
        dimension_for_model: the desired dimension of model as specified from the embeddings layer
        num_of_heads: the desired number of heads wanted from the multi-head-attention mechanism, also specified within encoders
        dim_feedforward: the dimension for the feedforward module, defaulted to 2048
        dropout: mechanism to remove model dependencies on other factors, defaulted to 0.1
        '''

        super().__init__()
        self.self_attn = nn.MultiheadAttention(dimension_for_model, num_of_heads, dropout=dropout) # Masked self - attention
        self.cross_attn = nn.MultiheadAttention(dimension_for_model, num_of_heads, dropout=dropout) # Encoder decoder attention
        self.ffn = nn.Sequential(
            nn.Linear(dimension_for_model, dim_feedforward), # Feeding forward
            nn.ReLU(),
            nn.Linear(dim_feedforward, dimension_for_model),
        )

        # Layer normalizations
        self.norm1 = nn.LayerNorm(dimension_for_model)
        self.norm2 = nn.LayerNorm(dimension_for_model)
        self.norm3 = nn.LayerNorm(dimension_for_model)
        # Dropouts
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt_t = tgt.transpose(0, 1)  
        memory_t = memory.transpose(0, 1)  
        
        # Masked self-attention
        _tgt = tgt
        tgt2, _ = self.self_attn(tgt_t, tgt_t, tgt_t, attn_mask=tgt_mask)
        tgt2 = tgt2.transpose(0, 1)  # Back to [batch_size, seq_len, hidden_dim]
        tgt = self.norm1(_tgt + self.dropout1(tgt2))

        # Cross-attention with encoder output
        _tgt = tgt
        tgt2, _ = self.cross_attn(tgt_t, memory_t, memory_t, attn_mask=memory_mask)
        tgt2 = tgt2.transpose(0, 1)  # Back to [batch_size, seq_len, hidden_dim]
        tgt = self.norm2(_tgt + self.dropout2(tgt2))

        # Feed-forward
        _tgt = tgt
        tgt2 = self.ffn(tgt)
        tgt = self.norm3(_tgt + self.dropout3(tgt2))

        return tgt

class Decoder(nn.Module):
    def __init__(self, vocab_size, dimension_for_model, num_layers, num_of_heads, dim_feedforward=2048, dropout=0.1, max_len=5000):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dimension_for_model) # Embeds the data
        self.pe    = PositionalEncoding(dimension_for_model, dropout=dropout, max_len=max_len)  # Encodes using sine and cosine functions for different positions
        self.layers = nn.ModuleList([
            DecoderLayer(dimension_for_model, num_of_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dimension_for_model)

    def forward(self, tgt_seq, memory, tgt_mask=None, memory_mask=None):
        x = self.embed(tgt_seq) * math.sqrt(self.embed.embedding_dim) # Embedding and masking
        x = self.pe(x)
        for layer in self.layers:  # Iterating through encoding layers
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return self.norm(x) # Layer normalization