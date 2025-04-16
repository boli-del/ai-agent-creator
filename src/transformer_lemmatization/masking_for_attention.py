import numpy as np 
import torch

def mask(size):
    '''
    A function for creating a look-ahead mask, ensuring that tokens won't see future tokens during the process of training
    through the creation of upper-triangular matrixes
    size: number of tokens within the sequence
    '''
    sq_mat = (1, size, size) #creating a square matrix filled with 1
    np.triu(np.ones(sq_mat), k=1) #turning the square matrix into an upper triangular matrix
    return torch.from_numpy(1 - mask)
