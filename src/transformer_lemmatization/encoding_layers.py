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
        dimension_for_network: the dimension needed for the 
        '''

        super.__init__() #initializing the parent class - 'neural-networks'
        self.expansion = nn.Linear(dimension_for_model, dimension_for_network) #expanding the original batch taken from the multi-head attention into newer ones with the desired dimensions
        self.dropout = nn.Dropout(dropout) #creating the dropout layer for improving the model's ability through testing and training by replacing specific rows and columns with 0s
        self.activation = nn.ReLU() #introducing non-linearity into the encoder and allowing models to represent values non-linearly
        self.reverse_expansion = nn.Linear(dimension_for_network, dimension_for_model) # reducing the dimension from expanded into original
    def forward(self, x):

        '''
        Applying the process of the positional feed_forward function
        x: the data which the positional feed forward is applied to
        '''
        parsed = self.dropout(self.activation(self.expansion(x)))
        return self.reverse_expansion(parsed)
#using the layer_normalization to add outputs back and then normalize the layer

class Layer_normalization (nn.Module):
    def __init__(self, dimension_for_model, dropout = 0.1):
        print()
    def forward(self, mask = None):
        print()


