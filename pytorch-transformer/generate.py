import torch
from torch.functional import F
from models.transformer import TransformerModel
from config import *


class SloganGenerator:
    def __init__(self, model_path, tokenizer):
        # Initialize the model and tokenizer
        self.model = TransformerModel(vocab_size, d_model, nhead, num_decoder_layers, 
                 dim_feedforward, max_seq_length).to(device)
        self.tokenizer = tokenizer
        # Load the weight into the model
        model_dict = torch.load(model_path, weights_only=True)   
        self.model.load_state_dict(model_dict)   
        

    def generate_slogan(self, start_sequence, max_length=20):
        self.model.eval() # check if works without to device
        input_sequence = torch.tensor(self.tokenizer.encode(start_sequence), dtype=torch.long).unsqueeze(0)

        generated_sequence = input_sequence.tolist()[0]

        for _ in range(max_length - len(start_sequence)): 
            input_tensor = torch.tensor(generated_sequence[:max_length], dtype=torch.long).unsqueeze(0).to(device)
            with torch.no_grad():
                output = self.model(input_tensor)
            next_token = torch.argmax(F.softmax(output[0, -1, :], dim=0)).item()
            generated_sequence.append(next_token)
            if next_token == 102: # EOS token (for BERT)
                break

        return ' '.join([self.tokenizer.decode(idx, skip_special_tokens=True)
                        for idx in generated_sequence])



generator = SloganGenerator('slogan_generator.pth', tokenizer)
print(generator.generate_slogan('what'))



