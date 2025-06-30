import torch
import math
import torch.nn as nn
# Including embeding layers for converting tokens into vector spaces
class Embeddings(nn.Module):
    '''
    The constructor for the embeddings class, initializing a look up table that corresponds each words in the vocabulary chain to a vector
    char: the amount of unique characters passed in
    dimension_for_model: the desired dimension of vector that's desired to pass the word to
    num_of_roles: the number of roles passed in
    '''
    def __init__ (self, char, dimension_for_model, num_of_roles = 2, max_turns = 16):
        # Initializing parent function
        super(Embeddings, self).__init__()
        #creating an embedding layer and parsing the words into the matrix and dimension corresponding to the input
        self.lut = nn.Embedding(char, dimension_for_model) #stores data into look up table
        self.lut_roles = nn.Embedding (num_of_roles, dimension_for_model) #creating look up table for the number of roles
        self.lut_turns = nn.Embedding (max_turns, dimension_for_model) #creating look up table for the number of turns
        self.dimension_for_model = dimension_for_model  #stores variable
        self.norm = nn.LayerNorm(dimension_for_model)  #defining normalization methods
    '''
    looks up the corresponding number from the look up table when numbers are passed in
    x: a tensor of token indices
    '''
    def forward(self, x, roles, turns):
        var = self.lut(x)  # Initialize the variable with the lookup table information of actual speaking content - parsed to words
        var = var + self.lut(roles)  # Adding information about roles into the tensor
        var = var + self.lut(turns) # Adding information about speaking turn into the tensor

        # Normalizing the tensors
        var = var*math.sqrt(self.dimension_for_model)
        var = self.norm(var)
        return var
    
if __name__ == '__main__':
    d_model = 512  # Desired model_dimension size definition
    
    # Instead of scoping in words, move down a scope for characters, which is unarguably more beneficial 
    characters = list("abcdefghijklmnopqrstuvwxyz ")

    # Create a mapping from each character to its index.
    char2idx = {char: idx for idx, char in enumerate(characters)}
    vocab = len(characters)  # The vocabulary size is the number of unique characters

    # Create a look-up table for each character(role/speaker) within the chat
    look_up_table_roles = {'system': 0, 'user': 1}

    # Example input string.
    input_str = "01 system: hello world"

    # Splitting the conversation, position and role information from a line
    position = int(input_str[0:2].strip())
    input_str = input_str[2:]
    conversation = input_str.split(':')[1].strip()
    role = input_str.split(':')[0].strip()

    # Convert the input string into a list of indices
    # This filters out any character not in the vocabulary
    # Convert the roles into reference ids using the look up table
    conversation_indices = [char2idx[char] for char in conversation if char in char2idx]
    position_indices = [position for char in conversation if char in char2idx]
    role_indices = [look_up_table_roles[role] for char in conversation if char in char2idx]

    # Create tensors from the lists of indices.
    # Here we treat it as a batch with one sequence.
    conversations = torch.LongTensor([conversation_indices])
    roles = torch.LongTensor([role_indices])
    positions = torch.LongTensor([position_indices])

    # Initialize the embedding layer using the character-level vocabulary size.
    emb = Embeddings(vocab, d_model)
    embr = emb(conversations, roles, positions)

    print("embr:", embr)

