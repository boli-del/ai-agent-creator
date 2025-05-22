#including neccessary libraires
import math
import torch.nn as nn
import torch

#using the positional feed-forward network to expand dimension for model
class position_wide_feed_forward(nn.Module):
    def __init__(self, dimension_for_model, dimension_for_network, dropout = 0.1):

        '''
        A Constructor for the positional feed forward network
        dimensin_for_model: the manually decided dimension that's used in the embeddings layer
        dimension_for_network: the dimension needed to expand the embedded results into
        dropout: optional dropout to wipe out specific columns and rows of the matrix to improve the model's abilities during training
        '''

        super().__init__() #initializing the parent class - 'neural-networks'
        self.expansion = nn.Linear(dimension_for_model, dimension_for_network) #expanding the original batch taken from the multi-head attention into newer ones with the desired dimensions
        self.apply_dropout = nn.Dropout(dropout) #creating the dropout layer for improving the model's ability through testing and training by replacing specific rows and columns with 0s
        self.activation = nn.ReLU() #introducing non-linearity into the encoder and allowing models to represent values non-linearly
        self.reverse_expansion = nn.Linear(dimension_for_network, dimension_for_model) # reducing the dimension from expanded into original
    def forward(self, x):

        '''
        Applying the process of the positional feed_forward function
        x: the data which the positional feed forward is applied to
        '''
        parsed = self.apply_dropout(self.activation(self.expansion(x)))
        return self.reverse_expansion(parsed)
#using the layer_normalization to add outputs back and then normalize the layer

class Residual_layer (nn.Module):
    def __init__(self, dimension_for_model, dropout = 0.1):
        '''
        A Constructor for the Residual and Normalization Layer
        dropout: optional dropout to wipe out specific columns and rows of the matrix to improve the model's abilities during training
        dimension_for_model: The desired dimension from the embeddings layer
        '''
        super().__init__()
        self.normalize = nn.LayerNorm(dimension_for_model)  #creating the layer normalization
        self.apply_dropout = nn.Dropout(dropout) 
    def forward(self, input_tensor, sublayer_tensor):
        '''
        input_tensor: the collection of tensor sum at the current stage
        sublayer_tensor: the tensor from the specific sublayer and still needed to be added
        '''
        result = self.apply_dropout(sublayer_tensor)+input_tensor  #adding two results together, since both are of same dimension to enforce the positional arguments, but also apply dropout to the new tensor being added
        return self.normalize(result) #return the normalized result
    


if __name__ == '__main__':
    inp = torch.tensor([[
       [1.0, 2.0, 3.0, 4.0],
       [0.5, 1.5, 2.5, 3.5],
       [4.0, 3.0, 2.0, 1.0]
    ]])

    # Instantiate with no dropout, intermediate size 8
    ffn = position_wide_feed_forward(dimension_for_model=4, dimension_for_network=8, dropout=0.0)

    # Run it
    out = ffn(inp)

    # Print to verify shape and nontrivial transform
    print("Input:", inp)
    print("Output:", out)
    print("Output shape:", out.shape)
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                         [4.0, 3.0, 2.0, 1.0],
                         [0.5, 1.5, 2.5, 3.5]]])
    
    # Dummy “sublayer” output to add
    sub = torch.tensor([[[0.1, 0.1, 0.1, 0.1],
                         [0.2, 0.2, 0.2, 0.2],
                         [0.3, 0.3, 0.3, 0.3]]])
    
    # Instantiate your residual+norm block (no dropout)
    layer = Residual_layer(dimension_for_model=4, dropout=0.0)
    
    # Run it
    out = layer(x, sub)
    
    # Print everything
    print("Input X:\n", x)
    print("\nSublayer output:\n", sub)
    print("\nResidual+Norm output:\n", out)
        


