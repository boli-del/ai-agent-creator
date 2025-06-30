import torch
import math
import torch.nn as nn

# Positional Encoding Layer
class PositionalEncoding(nn.Module):
    """
    Positional encoding layer for transformer models, with adjustments for lemmatization.
    In lemmatization tasks (especially with character-level inputs), sequences are typically
    much shorter. Therefore, max_len is set to a lower value to better match the expected input.
    
    Parameters:
      - dimension_for_model: Dimensionality of the embedding vectors.
      - dropout: Dropout probability used for regularization.
      - max_len: Maximum sequence length; lowered here (e.g., 256) since lemmatization sequences are short.
    """
    def __init__(self, dimension_for_model, dropout, max_len=256):
        # Initialize the parent module.
        super(PositionalEncoding, self).__init__()
        # Create a dropout layer.
        self.dropout = nn.Dropout(p=dropout)
        
        # Initialize a positional encoding matrix with shape (max_len, dimension_for_model).
        pos_enc_matrix = torch.zeros(max_len, dimension_for_model)
        
        # Create a column vector of positions: 0, 1, 2, ..., max_len-1.
        position = torch.arange(0, max_len).unsqueeze(1)
        
        # Calculate scaling terms for sine and cosine functions.
        div_term = torch.exp(torch.arange(0, dimension_for_model, 2) * -(math.log(10000.0) / dimension_for_model))
        
        # For even indices in the embedding dimensions, apply sine.
        pos_enc_matrix[:, 0::2] = torch.sin(position * div_term)
        # For odd indices, apply cosine.
        pos_enc_matrix[:, 1::2] = torch.cos(position * div_term)
        
        # Add an extra batch dimension for easier addition to input embeddings.
        pos_enc_matrix = pos_enc_matrix.unsqueeze(0)
        
        # Register the positional encoding matrix as a buffer so it's not updated by the optimizer.
        self.register_buffer('pe', pos_enc_matrix)

    def forward(self, x):
        """
        Add positional encodings to the input tensor.
        x: Tensor of shape [batch_size, sequence_length, dimension_for_model]
        """
        # Add the positional encodings to the input (slice to match the input sequence length)
        x = x + self.pe[:, :x.size(1)].detach()
        return self.dropout(x)


# Example usage for a lemmatization task
if __name__ == '__main__':
    d_model = 512       # Embedding dimension.
    dropout_rate = 0.1  # Dropout probability.
    max_len = 256       # Adjusted maximum sequence length for short lemmatization inputs.
    
    # Instantiate the positional encoding layer with a smaller max_len suitable for lemmatization.
    pos_encoder = PositionalEncoding(dimension_for_model=d_model, dropout=dropout_rate, max_len=max_len)
    
    # Create a dummy input: a batch of 2 sequences with length 20 (for example, character-level tokens).
    dummy_input = torch.randn(2, 20, d_model)
    
    # Apply the positional encoder.
    encoded_output = pos_encoder(dummy_input)
    
    print("Encoded output shape:", encoded_output.shape)