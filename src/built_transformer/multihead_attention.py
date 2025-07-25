import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, dimension_for_model, num_of_heads, dropout = 0.1):
        '''
        initializes multi-head attention module
        dimension_for_model: the same variable as the one in the embeddings, meaning the dimensionality of the embeddings
        num_heads:  the number of attention heads
        dropout: as explained in positional_encodings, the dropout rate, defaulted to 0.1
        '''
 
        # Initializing the parent function
        super(MultiHeadAttention, self).__init__()
        assert dimension_for_model % num_of_heads == 0, "dimension_for_model must be devisible by num_of_heads"

        self.num_of_heads = num_of_heads
        self.dimension_for_model = dimension_for_model
        self.d_k = dimension_for_model//num_of_heads  # This is the dimension for each head

        # Creating linear layers for seperating data into query, key, and value
        self.linear_query = nn.Linear(dimension_for_model, dimension_for_model)
        self.linear_key = nn.Linear(dimension_for_model, dimension_for_model)
        self.linear_value = nn.Linear(dimension_for_model, dimension_for_model)
        self.linear_out = nn.Linear(dimension_for_model, dimension_for_model)  # Added linear out

        # Adding dropout layer
        self.dropout = nn.Dropout(dropout)
        # Defining and applying softmax
        self.softmax = nn.Softmax(dim = -1)
    
    def forward(self, query, key, value, mask = None):
        '''
        Forward pass for multi-head attention.
        query: tensors with shape batch_size, sequence_length, dimension_for_model
        key: same as query
        value: same as query and key
        mask: a tensor that can be applied to attention scores
        '''
        batch_size = query.size(0)
        seq_len = query.size(1)

        # Projecting using linear layers
        Q = self.linear_query(query)
        K = self.linear_key(key)
        V = self.linear_value(value)

        # Splitting tensors into multiple heads
        Q = Q.view(batch_size, seq_len, self.num_of_heads, self.d_k).transpose(1,2)
        K = K.view(batch_size, seq_len, self.num_of_heads, self.d_k).transpose(1,2)
        V = V.view(batch_size, seq_len, self.num_of_heads, self.d_k).transpose(1,2)
        # Applying the attention calculation formula
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Attention weight conversion
        attn = self.softmax(scores)
        attn = self.dropout(attn)

        output = torch.matmul(attn, V)

        # Concatonating outputs for all heads
        output = output.transpose(1,2).contiguous().view(batch_size, seq_len, self.dimension_for_model)

        # Linear projection to combine all heads
        output = self.linear_out(output)

        return output, attn