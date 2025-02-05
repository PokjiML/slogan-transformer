import torch
import math
import re

def positional_encoding(seq_len, embed_dim):
    pe = torch.zeros(seq_len, embed_dim)
    for pos in range(seq_len):
        for i in range(0, embed_dim, 2):
            pe[pos, i] = math.sin(pos / (10000 ** (2 * i / embed_dim)))
            pe[pos, i + 1] = math.cos(pos / (10000 ** (2 * i / embed_dim)))
    return pe.unsqueeze(0)  # Output for batch_dim propagation

def generate_padding_mask(sequence, pad_token=0):
    mask = (sequence == pad_token).float()
    mask = mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
    return mask

def generate_look_ahead_mask(size):
    mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
    return mask

def clean_text(text):
    """Clean generated text"""

    # Remove space at the beginning
    text = re.sub(r'^\s+', '', text)
    # Remove double spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove spaces between '
    text = re.sub(r"\s'\s", "'", text)
    # Remove spaces between .
    text = re.sub(r"\s\.", ".", text)
    # Remove spaces between ,
    text = re.sub(r"\s,\s", ",", text)
    # Capitalize the first letter
    text = re.sub(r'^\w', lambda m: m.group().upper(), text)
    # Capitalize the letter after .
    text = re.sub(r'\.\s+(\w)', lambda m: m.group().upper(), text)
    # Merge words with ##
    text = re.sub(r' ##', '', text)

    return text