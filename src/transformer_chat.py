import torch
import torch.nn as nn
import math

# Import your building blocks
from src.built_transformer.embeddings import Embeddings
from src.built_transformer.encoder import Encoder, EncoderLayer
from src.built_transformer.decoders import Decoder, DecoderLayer
from src.built_transformer.positional_encodings import PositionalEncoding
from src.built_transformer.slot_classifier import SlotClassifier

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
            vocab_size=vocab_size,
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
            max_len=max_len
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
        # Combine embeddings
        x = self.embed(src_tokens, src_roles, src_turns)
        x = self.pos_enc(x)
        
        # Pass through encoder
        return self.encoder(src_tokens, src_mask)
    
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
