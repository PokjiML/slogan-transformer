import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

all_slogans = pd.read_csv('all_slogans.csv', sep=';')
slogans = all_slogans['slogan']
slogans = slogans.str.lower()

# reducing invaluable tokens
to_remove = ['\n', '\r', '>', '\x80', '\x93', '\x94', '\x99', '\x9d', '\xa0',
             '¦', '®', '°', 'º', '¼', '½','×', 'â', 'ã', 'è', 'é', 'ï', 'ñ', 'ú', 'ü',
             '⁄', '（', '）', '，', '·']

dict_to_remove = {"’" : "'", "‘" : "'", "“" : '"', "”" : '"',
                  "…" : '...', '—': '-', '–': '-'}


# normalizing the tokens
for char in to_remove:
    slogans = slogans.str.replace(char, ' ')


for key, value in dict_to_remove.items():
    slogans = slogans.str.replace(key, value)


# getting the character set
characters = [char for slogan in slogans for char in slogan]
characters = sorted((set(characters)))


# encoding string to integers sequence
# decoding integers to string sequence
to_int = {char: idx for idx, char in enumerate(characters)}
to_str = {idx: char for idx, char in enumerate(characters)}

encode = lambda sentence: [to_int[char] for char in sentence]
decode = lambda sentence: [to_str[char] for char in sentence]


# predict the next character
example = slogans[0]

for i in range(len(example)):
    print(f'Input: {example[:i+1]}')
    print(f'Output: {example[i+1:i+2]}')
    print()

