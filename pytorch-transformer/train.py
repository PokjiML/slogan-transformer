import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim                         # Adam optimizer
import torch.nn.functional as F                     # Softmax function
from torch.utils.data import DataLoader, Dataset    # Loading batches
from torch.optim.lr_scheduler import OneCycleLR     # Learning rate scheduler
from models.transformer import TransformerModel

from data.dataset import SloganDataset
from data.preprocess import load_and_clean_data, tokenize_slogans
from utils import generate_padding_mask
from config import *                                # Import hyperparameters


class Trainer:
    def __init__(self, model, dataloader, criterion, optimizer, scheduler, num_epochs=20):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs

    def train(self):
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch}')
            for batch in self.dataloader:
                input_sequences, target_sequences = batch
                input_sequences, target_sequences = batch
                input_sequences = input_sequences.to(device)
                target_sequences = target_sequences.to(device)

                self.optimizer.zero_grad()
                output = self.model(input_sequences)
                loss = self.criterion(output.view(-1, vocab_size), target_sequences.view(-1))
                loss.backward()

                self.optimizer.step()
                self.scheduler.step()

            print(f'Epoch {epoch} Loss: {loss.item()}, LR: {self.scheduler.get_last_lr()[0]:.6f}')


if __name__ == '__main__':

    # Load and preprocess data
    slogans = load_and_clean_data('all_slogans.csv')
    encoded_slogans = tokenize_slogans(slogans, 'bert-base-uncased')

    # Create dataset and dataloader
    dataset = SloganDataset(encoded_slogans)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Initialize model, criterion, optimizer, and scheduler
    model = TransformerModel(vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward, max_seq_length).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = OneCycleLR(optimizer, max_lr=0.0001, epochs=20, steps_per_epoch=len(dataloader))

    # Train the model
    trainer = Trainer(model, dataloader, criterion, optimizer, scheduler)
    trainer.train()

    # Save the model after training
    torch.save(model.state_dict(), 'slogan_generator.pth')


