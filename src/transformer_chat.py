import torch
import torch.nn as nn
import math

# Import neccessary layers
from built_transformer.embeddings import Embeddings
from built_transformer.encoder import Encoder, EncoderLayer
from built_transformer.decoders import Decoder, DecoderLayer
from built_transformer.positional_encodings import PositionalEncoding
from built_transformer.slot_classifier import SlotClassifier

class TransformerChatbot(nn.Module):
    """
    Unified Transformer-based chatbot model that combines:
    - Joint token/role/turn embeddings
    - Encoder-decoder architecture with attention
    - Slot-filling classification
    - Generation capabilities
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_roles: int = 2,
        max_turns: int = 16,
        num_slots: int = 4,
        dropout: float = 0.1,
        max_len: int = 5000
    ):
        super().__init__()
        
        # Embeddings for tokens, roles, and turns
        self.embed = Embeddings(
            char=vocab_size, # Fixed type and name mismatch
            dimension_for_model=d_model,
            num_of_roles=num_roles,
            max_turns=max_turns
        )
        
        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len)
        
        # Encoder stack
        self.encoder = Encoder(
            vocab_size=vocab_size,
            dimension_of_model=d_model,
            num_of_heads=num_heads,
            num_layers=num_encoder_layers,
            dim_feedforward=d_ff,
            dropout=dropout,
            max_len=max_len,
            num_of_roles=num_roles,
            max_turns=max_turns
        )
        
        # Decoder stack
        self.decoder = Decoder(
            vocab_size=vocab_size,
            dimension_for_model=d_model,
            num_layers=num_decoder_layers,
            num_of_heads=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            max_len=max_len
        )
        
        # Output projections
        self.out_proj = nn.Linear(d_model, vocab_size)
        self.slot_classifier = SlotClassifier(d_model, num_slots)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        #Initialize parameters with Xavier uniform initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def load_state_dict(self, state_dict, strict=True):
        # Check if this is an old model format (has encoder.embed.weight), since previous versions uses different weights
        if 'encoder.embed.weight' in state_dict:
            # This is an old model, we need to adapt the weights
            old_embed_weight = state_dict['encoder.embed.weight']
            
            # Copy the old embedding weights to the new structure
            state_dict['encoder.embed.lut.weight'] = old_embed_weight
            # Initialize role and turn embeddings with correct sizes
            state_dict['encoder.embed.lut_roles.weight'] = torch.zeros(2, old_embed_weight.size(1))  # 2 roles
            state_dict['encoder.embed.lut_turns.weight'] = torch.zeros(16, old_embed_weight.size(1))  # 16 turns
            state_dict['encoder.embed.norm.weight'] = torch.ones(old_embed_weight.size(1))
            state_dict['encoder.embed.norm.bias'] = torch.zeros(old_embed_weight.size(1))
            
            # Remove the old key
            del state_dict['encoder.embed.weight']
        
        return super().load_state_dict(state_dict, strict=strict)
                
    def encode(self, src_tokens, src_roles, src_turns, src_mask=None):
        """
        Encode source sequences with role and turn information.
        Args:
            src_tokens: [B, S] token IDs
            src_roles:  [B, S] role IDs
            src_turns:  [B, S] turn IDs
            src_mask:   padding mask [B, 1, 1, S]
        Returns:
            enc_out: [B, S, d_model]
        """
        # Pass through encoder (embedding and positional encoding handled inside)
        return self.encoder(src_tokens, src_roles, src_turns, src_mask)
    
    def decode(
        self,
        tgt_tokens,
        enc_out,
        tgt_roles,
        tgt_turns,
        src_mask=None,
        tgt_mask=None
    ):
        """
        Decode target sequences with encoder context.
        Args:
            tgt_tokens: [B, T] target token IDs
            enc_out:    [B, S, d_model] encoder output
            tgt_roles:  [B, T] target role IDs
            tgt_turns:  [B, T] target turn IDs
            src_mask:   [B, 1, 1, S] source mask
            tgt_mask:   [B, 1, T, T] target mask
        Returns:
            logits: [B, T, vocab_size]
        """
        # Combine embeddings
        y = self.embed(tgt_tokens, tgt_roles, tgt_turns)
        y = self.pos_enc(y)
        
        # Pass through decoder
        dec_out = self.decoder(tgt_tokens, enc_out, tgt_mask, src_mask)
        return self.out_proj(dec_out)
    
    def forward(
        self,
        src_tokens,
        tgt_tokens,
        src_roles,
        tgt_roles,
        src_turns,
        tgt_turns,
        src_mask=None,
        tgt_mask=None
    ):
        """
        Full forward pass combining encoding, decoding, and slot classification.
        Args:
            src_tokens: [B, S] source token IDs
            tgt_tokens: [B, T] target token IDs
            src_roles:  [B, S] source role IDs
            tgt_roles:  [B, T] target role IDs
            src_turns:  [B, S] source turn IDs
            tgt_turns:  [B, T] target turn IDs
            src_mask:   [B, 1, 1, S] source mask
            tgt_mask:   [B, 1, T, T] target mask
        Returns:
            gen_logits: [B, T, vocab_size] generation logits
            slot_logits: [B, num_slots] slot classification logits
        """
        # Encode source sequence
        enc_out = self.encode(src_tokens, src_roles, src_turns, src_mask)
        
        # Decode target sequence
        gen_logits = self.decode(
            tgt_tokens,
            enc_out,
            tgt_roles,
            tgt_turns,
            src_mask,
            tgt_mask
        )
        
        # Use first position of encoder output for slot classification
        cls_rep = enc_out[:, 0, :]
        slot_logits = self.slot_classifier(cls_rep)
        
        return gen_logits, slot_logits
