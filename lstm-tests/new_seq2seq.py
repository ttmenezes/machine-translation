from glob import glob
from multiprocessing import pool
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import vocab
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint

### pre-processing (tokenization + vocabulary construction) ---------------------

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

ger_vocab = vocab(ger_counter, min_freq=1, specials=('<sos>','<eos>','<unk>'))
ger_vocab.set_default_index(ger_vocab['<unk>'])
eng_vocab = vocab(eng_counter, min_freq=1, specials=('<sos>','<eos>','<unk>'))
eng_vocab.set_default_index(eng_vocab['<unk>'])

# print(len(ger_vocab))
# print('index of das', ger_vocab.get_stoi()['das'])
# print('token at index 52', ger_vocab.get_itos()[52])
# print(len(eng_vocab))
# print('token at index 52', eng_vocab.get_itos()[52])

eng_text_transform = lambda x: [eng_vocab['<sos>']] + [eng_vocab[token] for token in tokenizer_eng(x)] + [eng_vocab['<eos>']]
ger_text_transform = lambda x: [ger_vocab['<sos>']] + [ger_vocab[token] for token in tokenizer_ger(x)] + [ger_vocab['<eos>']]
# testing ---
# eng_text_transform = lambda x: ['<sos>'] + [token for token in tokenizer_eng(x)] + ['<eos>']
# ger_text_transform = lambda x: ['<sos>'] + [token for token in tokenizer_ger(x)] + ['<eos>']
# end testing ---


# print('output', eng_text_transform('this is a test'))

### model building --------------------------------------------------------------


# first LSTM
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    # input: vector of word indices in vocabulary
    def forward(self, x):
        # x shape: (seq_length, N)
        
        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding)

        return hidden, cell


# second LSTM
class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super(Decoder, self).__init__()
        # hidden size is same for encoder and decoder
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden, cell):
        # shape of x: (N) but we want (1, N) as the decoder translates 1 word at a time,
        # given previous predicted word and previous hidden cell
        x = x.unsqueeze(0)

        self.dropout(self.embedding(x))

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        # shape of outputs: (1, N, hidden_size)

        predictions = self.fc(outputs)
        # shape of predictions: (1, N, length_of_output_vocab)

        predictions = predictions.squeeze(0)

        return predictions, hidden, cell

# combines encoder and decoder
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(eng_vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to("cpu")

        hidden, cell = self.encoder(source)

        # grab start token
        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)

            outputs[t] = output

            # output: (N, english_vocab_size), so argmax gets index of best guess from decoder
            best_guess = output.argmax(1)

            x = target[t] if random.random() < teacher_force_ratio else best_guess
        
        return outputs


### model training --------------------------------------------------------------

## model parameters ---
# train hyperparameters
num_epochs = 20
learning_rate = 0.001
batch_size = 64

# model hyperparameters
load_model = True
device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size_encoder = len(ger_vocab)
input_size_decoder = len(eng_vocab)
output_size = len(eng_vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5

# Tensorboard
writer = SummaryWriter(f'runs/loss_plot')
step = 0

## build batches of similar lengths of text ---

def collate_batch(batch):
    ger_list, eng_list = [], []
    for (_src, _target) in batch:
        ger_list.append(torch.tensor(ger_text_transform(_src)))
        eng_list.append(torch.tensor(eng_text_transform(_target)))

    return pad_sequence(ger_list, padding_value=3.0), pad_sequence(eng_list, padding_value=3.0)

train_iter = iter(Multi30k(split='train'))
train_list = list(train_iter)
batch_size = 8

# understand this!
def batch_sampler():
    indices = [(i, len(tokenizer_ger(s[1]))) for i, s in enumerate(train_list)]
    random.shuffle(indices)
    pooled_indices = []
    # create pool of indices of similar lengths
    for i in range(0, len(indices), batch_size*100):
        pooled_indices.extend(sorted(indices[i:i + batch_size*100], key=lambda x:x[1]))

    pooled_indices = [x[0] for x in pooled_indices]

    for i in range(0, len(pooled_indices), batch_size):
        yield pooled_indices[i:i + batch_size]


train_dataloader = DataLoader(train_list, batch_sampler=batch_sampler(), collate_fn=collate_batch)

# for idx, (ger,eng) in enumerate(train_dataloader):
#     print(ger)
#     print('\n\n')

## init model ---

encoder_net = Encoder(input_size_encoder, encoder_embedding_size, 
                        hidden_size, num_layers, enc_dropout).to(device)

decoder_net = Decoder(input_size_decoder, decoder_embedding_size, 
                        hidden_size, output_size, num_layers, dec_dropout).to(device)

model = Seq2Seq(encoder_net, decoder_net).to('cpu')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# pad_idx = eng_vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=3)

if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.tar'), model, optimizer)

translated_example = translate_sentence(
        model, 
        'Der Professor ist nicht nett',
        ger_vocab,
        eng_vocab,
        device,
        max_length=50
        )
print(f'translated example:', translated_example)

# for epoch in range(num_epochs):
#     print(f'Epoch {epoch} / {num_epochs}')

#     checkpoint = {'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()}
#     save_checkpoint(checkpoint)

#     # see translation progress over time
#     # model.eval()
#     translated_example = translate_sentence(
#         model, 
#         'zwei Personen rennen in den Laden',
#         ger_vocab,
#         eng_vocab,
#         device,
#         max_length=50
#         )
    
#     # print(f'translated example:', translated_example)
#     # model.train()
#     # ---

#     for batch_idx, (ger_batch, eng_batch) in enumerate(train_dataloader):
#         inp_data = ger_batch.to("cpu")
#         target = eng_batch.to("cpu")

#         output = model(inp_data, target).to("cpu")
#         # output shape: (trg_len, batch_size, output_dim)

#         output = output[1:].reshape(-1, output.shape[2])
#         target = target[1:].reshape(-1)

#         optimizer.zero_grad()
#         loss = criterion(output, target)

#         loss.backward()

#         # prevent exploding gradients
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

#         optimizer.step()

#         writer.add_scalar('Training Loss', loss, global_step=step)
#         step += 1
