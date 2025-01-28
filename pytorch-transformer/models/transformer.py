import torch.nn as nn
import math
from utils import positional_encoding, generate_look_ahead_mask, generate_padding_mask
from config import *


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_decoder_layers, 
                 dim_feedforward, max_seq_length):
        super(TransformerModel, self).__init__()
        
        # Create the token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Initialize weights with Xavier normal for stability
        nn.init.xavier_normal_(self.embedding.weight) 

        # Unsqueeze to add batch dimension
        self.pos_encoder = positional_encoding(max_seq_length, d_model).to(device)

        # Transformer Decoder layers
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(
            self.transformer_decoder_layer, num_layers=num_decoder_layers
        )

        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, src):
        # Generate look ahead mask to prevent looking at future tokens
        tgt_mask = generate_look_ahead_mask(src.size(1)).to(device) # check the change to 1
        # Use padding mask to prevent looking at not used tokens
        src_pad_mask = generate_padding_mask(src).to(device)
        # sqrt for stabilization
        src = self.embedding(src) * math.sqrt(d_model) # (batch_size, seq_len, d_model)
        # add positional encoding 
        src = src + self.pos_encoder[:, :src.size(1), :] # src.size(1) = seq_len
        output = self.transformer_decoder(tgt=src, memory=src, tgt_mask=tgt_mask,
                                          memory_mask=tgt_mask, tgt_key_padding_mask=src_pad_mask) # Change the memory mask
        output = self.dropout(output)
        output = self.fc_out(output)
        
        return output