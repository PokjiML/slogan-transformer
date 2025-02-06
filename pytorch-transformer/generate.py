import torch
from torch.functional import F
from models.transformer import TransformerModel
from config import *
from utils import clean_text
from models.get_model import download_model

# Download the model state_dict
# download_model()              <- Pre-trained model

class SloganGenerator:
    def __init__(self, model_path, tokenizer):
        # Initialize the model and tokenizer
        self.model = TransformerModel(vocab_size, d_model, nhead, num_decoder_layers, 
                 dim_feedforward, max_seq_length).to(device)
        self.tokenizer = tokenizer
        # Load the weight into the model
        model_dict = torch.load(model_path, weights_only=True)   
        self.model.load_state_dict(model_dict)   
        

    def generate_slogan(self, start_sequence, max_length=20, temperature=0.1):
        self.model.eval() # check if works without to device
        input_sequence = torch.tensor(self.tokenizer.encode(start_sequence), dtype=torch.long).unsqueeze(0)

        generated_sequence = input_sequence.tolist()[0]

        for _ in range(max_length - len(start_sequence)): 
            input_tensor = torch.tensor(generated_sequence[:max_length], dtype=torch.long).unsqueeze(0).to(device)
            with torch.no_grad():
                output = self.model(input_tensor)
            # Predict next token
            logits = output[0, -1, :] / temperature
            probabilities = F.softmax(logits, dim=0)
            next_token = torch.multinomial(probabilities, 1).item()

            generated_sequence.append(next_token)
            if next_token == 102: # EOS token (for BERT)
                break

        return ' '.join([self.tokenizer.decode(idx, skip_special_tokens=True)
                        for idx in generated_sequence])


# Generate 10 different slogans
for _ in range(20):
    generator = SloganGenerator('slogan_generator.pth', tokenizer)
    gen_seq = generator.generate_slogan('fly ', temperature=0.5)
    print(clean_text(gen_seq))


