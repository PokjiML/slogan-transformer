from torch.utils.data import Dataset

class SloganDataset(Dataset):
    def __init__(self, encoded_slogans, max_seq_length=20):
        self.encoded_slogans = encoded_slogans
        self.max_seq_length = max_seq_length
        
    def __len__(self):
        return len(self.encoded_slogans)
    
    def __getitem__(self, idx):
        slogan = self.encoded_slogans[idx]
        
        # Truncate if slogan is too long
        if len(slogan) > self.max_seq_length:
            slogan = slogan[:self.max_seq_length]     
        
        input_sequence = slogan[:-1]
        target_sequence = slogan[1:]
        return input_sequence, target_sequence