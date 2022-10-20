from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint

spacy_ger = spacy.load('de')
spacy_eng = spacy.load('en')

def tokenizer_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

# pre-processing
german = Field(tokenize=tokenizer_ger, lower=True,
                init_token='<sos>', eos_token='<eos>')

english = Field(tokenize=tokenizer_eng, lower=True,
                init_token='<sos>', eos_token='<eos>')

# train_data, validation_data, test_data = Multi30k(split=('train','validation','test'), language_pair=('de','en'))
train_data, validation_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                        fields=(german, english))



# build vocabulary based on words found in dataset
german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

# create model

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
        target_vocab_size = len(english.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size)

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

    
### now we train the model

# train hyperparameters
num_epochs = 20
learning_rate = 0.001
batch_size = 64

# model hyperparameters
load_model = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5

# Tensorboard
writer = SummaryWriter(f'runs/loss_plot')
step = 0

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, validation_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key = lambda x: len(x.src),
    device=device
)

encoder_net = Encoder(input_size_encoder, encoder_embedding_size, 
                        hidden_size, num_layers, enc_dropout).to(device)

decoder_net = Decoder(input_size_decoder, decoder_embedding_size, 
                        hidden_size, output_size, num_layers, dec_dropout).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = english.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.ptar'), model, optimizer)

for epoch in range(num_epochs):
    print(f'Epoch {epoch} / {num_epochs}')

    checkpoint = {'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()}
    save_checkpoint(checkpoint)

    for batch_idx, batch in enumerate(train_iterator):
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        output = model(inp_data, target)
        # output shape: (trg_len, batch_size, output_dim)

        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)

        loss.backward()

        # prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()

        writer.add_scalar('Training Loss', loss, global_step=step)
        step += 1