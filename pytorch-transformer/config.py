import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
vocab_size = tokenizer.vocab_size
d_model = 384  # Dimension of the embedding vector
nhead = 8  # Number of attention heads
num_decoder_layers = 3  # Number of decoder layers
dim_feedforward = 2048  # Feed-forward network dimension
max_seq_length = 20  # Maximum sequence length
batch_size = 128
dropout = 0.1
PAD_TOKEN = tokenizer.pad_token_id