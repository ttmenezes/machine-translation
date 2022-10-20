from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import vocab
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint

# data splits
train_data, validation_data, test_data = Multi30k(split=('train','valid','test'), language_pair=('de','en'))

train_iter = iter(train_data)
validation_iter = iter(validation_data)
test_iter = iter(test_data)

tokenizer_ger = get_tokenizer(tokenizer='spacy', language='de_core_news_sm')
tokenizer_eng = get_tokenizer(tokenizer='spacy', language='en_core_web_sm')

ger_counter = Counter()
eng_counter = Counter()
for (ger_line, eng_line) in train_iter:
    ger_counter.update(tokenizer_ger(ger_line))
    eng_counter.update(tokenizer_eng(eng_line))

ger_vocab = vocab(ger_counter, min_freq=2, specials=('<sos>','<eos>'))
eng_vocab = vocab(eng_counter, min_freq=2, specials=('<sos>','<eos>'))

print(len(ger_vocab))
# print('index of das', ger_vocab.get_stoi()['das'])
print('token at index 52', ger_vocab.get_itos()[52])
print(len(eng_vocab))
print('token at index 52', eng_vocab.get_itos()[52])
