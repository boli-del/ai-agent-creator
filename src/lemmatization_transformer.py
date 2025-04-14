import torch
import math
import torch.nn as nn
#including embeding layers for converting tokens into vector spaces
class Embeddings(nn.Module):
    '''
    vocab: the amount of unique words passed in
    dimension_for_model: the desired dimension of vector that's desired to pass the word to
    '''
    def _init_ (self, vocab, dimension_for_model):
        #initializing parent function
        super(Embeddings, self).__init__()
        #creating an embedding layer and parsing the words into the matrix and dimension corresponding to the input
        self.lut = nn.Embedding(vocab, dimension_for_model) #stores data into look up table
        self.dimension_for_model = dimension_for_model  #stores variable
    '''
    x: a tensor of token indices
    '''
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.dimension_for_model)  # returns the 
    
if __name__ == '__main__':
    d_model = 512
    vocab = 1000
    x = torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]])
    emb = Embeddings(vocab, d_model)
    embr = emb(x)
    print('embr:', embr)

#including positional encoding so that model understands sequence order
#create encoder (multi-head attention, feed-forward sub-layers)
#decoder(generates the output sequence)
#output projection