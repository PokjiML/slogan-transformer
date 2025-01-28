import pandas as pd
from transformers import AutoTokenizer


# Remove unwanted characters from slogans
def load_and_clean_data(file_path: str):
    all_slogans = pd.read_csv(file_path, sep=';')
    slogans = all_slogans['slogan'].str.lower()

    to_remove = ['\\n', '\\r', '>', '\\x80', '\\x93', '\\x94', '\\x99', '\\x9d', '\\xa0',
                 '¦', '®', '°', 'º', '¼', '½', '×', 'â', 'ã', 'è', 'é', 'ï', 'ñ', 'ú', 'ü',
                 '⁄', '（', '）', '，', '·']

    dict_to_remove = {"’": "'", "‘": "'", "“": '"', "”": '"',
                      "…": '...', '—': '-', '–': '-'}

    for char in to_remove:
        slogans = slogans.str.replace(char, ' ')

    for key, value in dict_to_remove.items():
        slogans = slogans.str.replace(key, value)

    return slogans.tolist()


# Tokenize slogans
def tokenize_slogans(slogans, tokenizer_name, max_length = 20):
    # Load tokenizer from transformers library
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # Initialize tokenizer
    encoded_slogans = tokenizer.batch_encode_plus(
        slogans,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    # Return only useful part (tokens)
    return encoded_slogans['input_ids']