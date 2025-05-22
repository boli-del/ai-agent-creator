from datasets import load_dataset
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
import torch
from torch.utils.data import dataloader
import tokenizers
raw_dataset = load_dataset("tuetscheck/atis", split = "train") #loading the atis dataset for intention training
tokenizer = Tokenizer.from_file('tokenizer.json') #using the tokenizer that was pretrained and set into json file

def tokenize(examples):
    return tokenizer(examples['text'], padding = 'max_length', truncation = True)
