import torch
import torch.nn as nn
import torch.optim as optim                         # Adam optimizer
import torch.nn.functional as F                     # Softmax function
from torch.utils.data import DataLoader, Dataset    # Loading batches
import torch.nn.utils.rnn as rnn_utils              # Padding the sequence
import pandas as pd
import numpy as np
import math

# For GPU training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Load the data
all_slogans = pd.read_csv('all_slogans.csv', sep=';')
slogans = all_slogans['slogan']
slogans = slogans.str.lower()

# reducing invaluable tokens
to_remove = ['\n', '\r', '>', '\x80', '\x93', '\x94', '\x99', '\x9d', '\xa0',
             '¦', '®', '°', 'º', '¼', '½','×', 'â', 'ã', 'è', 'é', 'ï', 'ñ', 'ú', 'ü',
             '⁄', '（', '）', '，', '·']

dict_to_remove = {"’" : "'", "‘" : "'", "“" : '"', "”" : '"',
                  "…" : '...', '—': '-', '–': '-'}


# removing useless toknes
for char in to_remove:
    slogans = slogans.str.replace(char, ' ')

# replacing tokens with normalised versions
for key, value in dict_to_remove.items():
    slogans = slogans.str.replace(key, value)


# getting the characters (tokens) set
characters = [char for slogan in slogans for char in slogan]
characters = sorted((set(characters)))


# adding in the end of every slogan 'E' end token
slogans = slogans + 'E'
characters = ['E'] + characters

# adding the start of sequence token 'S'
slogans = slogans.apply(lambda x: 'S' + x)
characters = ['S'] + characters

# Add padding token at 0 index
characters = ['P'] + characters


# encoding string to integers sequence
# decoding integers to string sequence
to_int = {char: idx for idx, char in enumerate(characters)}
to_str = {idx: char for idx, char in enumerate(characters)}

encode = lambda sentence: [to_int[char] for char in sentence]
decode = lambda sentence: [to_str[char] for char in sentence]

encoded_slogans = [encode(slogan) for slogan in slogans]


### Loading Data


class SloganDataset(Dataset):
    def __init__(self, slogans, encode, max_seq_length=100):
        self.slogans = slogans
        self.encode = encode
        self.max_seq_length = max_seq_length
        
    def __len__(self):
        return len(self.slogans)
    
    def __getitem__(self, idx):
        slogan = self.slogans[idx]
        
        # Truncate if slogan is too long
        if len(slogan) > self.max_seq_length:
            slogan = slogan[:self.max_seq_length]     

        input_sequence = torch.tensor(self.encode(slogan[:-1]), dtype=torch.long)
        target_sequence = torch.tensor(self.encode(slogan[1:]), dtype=torch.long)
        return input_sequence, target_sequence
    

# padding the sequence (For the largest in batch)
def collate_fn(batch): 
    input_sequences, target_sequences = zip(*batch)
    input_sequences_padded = rnn_utils.pad_sequence(input_sequences, batch_first=True, padding_value=0)
    target_sequences_padded = rnn_utils.pad_sequence(target_sequences, batch_first=True, padding_value=0)
    return input_sequences_padded, target_sequences_padded


# Test with subset of slogans
subset_slogans = slogans[:500]
dataset = SloganDataset(subset_slogans, encode)
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


### MASKS AND POSITIONAL ENCODING


# Sinusoidal positional encoding
def positional_encoding(seq_len, embed_dim):
    pe = torch.zeros(seq_len, embed_dim)
    for pos in range(seq_len):
        for i in range(0, embed_dim, 2):
            pe[pos, i] = math.sin(pos / (10000 ** (2 * i / embed_dim)))
            pe[pos, i + 1] = math.cos(pos / (10000 ** (2 * i / embed_dim)))
    return pe

# Generate padding mask to prevent looking at not used tokens
def generate_padding_mask(sequence, pad_token = 0):
    return (sequence != pad_token).unsqueeze(1).unsqueeze(2)

# Generate look ahead mask to prevent looking at future tokens
def generate_look_ahead_mask(size):
    mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
    return mask


### MODEL test


# define hyperparameters
vocab_size = len(characters)
d_model = 512 # dim of the embedding vector
nhead = 8 # number of attention heads
num_encoder_layers = 3 # number of encoder layers
num_decoder_layers = 3 # number of decoder layers
dim_feedforward = 2048 # feed-forward network dimension
max_seq_length = 100 
dropout = 0.1
PAD_TOKEN = 0


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_decoder_layers, 
                 dim_feedforward, max_seq_length):
        super(TransformerModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        # unsqueeze to add batch dimension
        self.pos_encoder = positional_encoding(max_seq_length, d_model).unsqueeze(0).to(device)

        # Transformer Decoder layers
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.transformer_decoder_layer, num_layers=num_decoder_layers
        )

        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, src):
        tgt_mask = generate_look_ahead_mask(src.size(1)).to(src.device) # check the change to 1
        # sqrt for stabilization
        src = self.embedding(src) * math.sqrt(d_model) # (batch_size, seq_len, d_model)
        # add positional encoding 
        src = src + self.pos_encoder[:, :src.size(1), :] # src.size(1) = seq_len
        output = self.transformer_decoder(tgt=src, memory=src, tgt_mask=tgt_mask)
        output = self.dropout(output)
        output = self.fc_out(output)
        
        return output
    
model = TransformerModel(vocab_size, d_model, nhead, 
                          num_decoder_layers, dim_feedforward, max_seq_length).to(device) # Watch out



criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
optimizer = optim.Adam(model.parameters(), lr=0.001)



### MODEL TRAINING

# Example training loop with dataloader
num_epochs = 10
for epoch in range(num_epochs):
    print(f'Epoch {epoch}')
    for batch in dataloader:
        # Move to GPU
        input_sequences, target_sequences = batch
        input_sequences = input_sequences.to(device)
        target_sequences = target_sequences.to(device)
        optimizer.zero_grad()
        output = model(input_sequences)
        loss = criterion(output.view(-1, vocab_size), target_sequences.view(-1))
        loss.backward()
        optimizer.step()

        
    print(f'Epoch: {epoch}, Loss: {loss.item()}')

# Text generation

def generate_slogan(model, start_sequence, max_length=100):
    model.eval()
    input_sequence = torch.tensor(encode(start_sequence), dtype=torch.long).unsqueeze(0)
    generated_sequence = input_sequence.tolist()[0]

    for _ in range(max_length - len(start_sequence)):   # Watch out
        input_tensor = torch.tensor(generated_sequence[-max_length:], dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
        next_token = torch.argmax(F.softmax(output[0, -1, :], dim=0)).item()
        generated_sequence.append(next_token)
        if to_str[next_token] == 'E':
            break
    
    return ''.join([to_str[idx] for idx in generated_sequence])

start_sequence = "th"
generated_slogan = generate_slogan(model, start_sequence)
print(f"Generated slogan: {generated_slogan}")