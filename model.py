import torch
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
print(characters)
len(characters)