#integrating the neccessary classes
import torch, torch.nn as nn, math, numpy as np
from encoding_layers import position_wide_feed_forward, Residual_layer
from masking_for_attention import mask
from multihead_attention import MultiHeadAttention 

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, dimension_for_model, num_of_heads, )