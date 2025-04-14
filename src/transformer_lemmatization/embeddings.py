import torch
import math
import torch.nn as nn
#including embeding layers for converting tokens into vector spaces
class Embeddings(nn.Module):
    '''
    The constructor for the embeddings class, initializing a look up table that corresponds each words in the vocabulary chain to a vector
    char: the amount of unique characters passed in
    dimension_for_model: the desired dimension of vector that's desired to pass the word to
    '''
    def __init__ (self, char, dimension_for_model):
        #initializing parent function
        super(Embeddings, self).__init__()
        #creating an embedding layer and parsing the words into the matrix and dimension corresponding to the input
        self.lut = nn.Embedding(char, dimension_for_model) #stores data into look up table
        self.dimension_for_model = dimension_for_model  #stores variable
    '''
    looks up the corresponding number from the look up table when numbers are passed in
    x: a tensor of token indices
    '''
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.dimension_for_model)  # multiplying by the square root of the dimension to manage the size of the numbers
    
if __name__ == '__main__':
    d_model = 512  #desired model_dimension size definition
    
    # instead of scoping in words, move down a scope for characters, which is unarguably more beneficial 
    characters = list("abcdefghijklmnopqrstuvwxyz ")

    # Create a mapping from each character to its index.
    char2idx = {char: idx for idx, char in enumerate(characters)}
    vocab = len(characters)  # The vocabulary size is the number of unique characters.

    # Example input string.
    input_str = "hello world"

    # Convert the input string into a list of indices (one per character).
    # This filters out any character not in our vocabulary.
    indices = [char2idx[char] for char in input_str if char in char2idx]

    # Create a tensor from the list of indices.
    # Here we treat it as a batch with one sequence.
    x = torch.LongTensor([indices])

    # Initialize the embedding layer using the character-level vocabulary size.
    emb = Embeddings(vocab, d_model)
    embr = emb(x)

    print("embr:", embr)

#including positional encoding so that model understands sequence order
#create encoder (multi-head attention, feed-forward sub-layers)
#decoder(generates the output sequence)
#output projection