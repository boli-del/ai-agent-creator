from datasets import load_dataset
from tokenizers import (
    Tokenizer, models, normalizers, pre_tokenizers,
    decoders, processors, trainers
)

#Load data and prepare iterator
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i:i+1000]['text']
#Initialize tokenizer and core components
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFD(),
    normalizers.Lowercase(),
    normalizers.StripAccents()
])
pre_tok = pre_tokenizers.Sequence([
    pre_tokenizers.WhitespaceSplit(),
    pre_tokenizers.Punctuation()
])
tokenizer.pre_tokenizer = pre_tok
tokenizer.decoder = decoders.WordPiece(prefix="##")
#Enable padding & truncation (works once [PAD] is in vocab post-train too)
tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=128)
tokenizer.enable_truncation(max_length=128)
#Train the tokenizer
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
#fetch special token IDs
cls_id = tokenizer.token_to_id("[CLS]")
sep_id = tokenizer.token_to_id("[SEP]")
#Attach post-processor with a dict mapping
tokenizer.post_processor = processors.TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B [SEP]",
    special_tokens=[         ("[CLS]", cls_id),
         ("[SEP]", sep_id)
     ]
)

#Save and test
tokenizer.save("tokenizer.json")
encoding = tokenizer.encode("Let's test this tokenizer.")
print(encoding.tokens)
