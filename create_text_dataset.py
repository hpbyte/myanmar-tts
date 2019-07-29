import pandas as pd
from sklearn.externals import joblib
from constants import *
from processing.proc_text import transform_text

metadata = pd.read_csv('data/LJSpeech/metadata.csv', sep='|', dtype='object', quoting=3, header=None)

metadata = metadata.iloc[:500]

metadata['norm_lower'] = metadata[2].apply(lambda x: x.lower())

texts = metadata['norm_lower']

# infer the vocabulary
existing_chars = list(set(texts.str.cat(sep=' ')))
vocab = ''.join(existing_chars)
vocab += 'P'  # add padding character

# Create association between vocab and id
vocab_id = {}
i = 0
for char in list(vocab):
  vocab_id[char] = i
  i += 1

text_input = transform_text(texts.values, vocab_id, NB_CHARS_MAX)

# split into training and testing
len_train = int(TRAIN_SET_RATIO * len(metadata))
text_input_training = text_input[:len_train]
text_input_testing = text_input[len_train:]

# save data
joblib.dump(text_input_training, 'data/LJSpeech/text_input_training.pkl')
joblib.dump(text_input_testing, 'data/LJSpeech/text_input_testing.pkl')

joblib.dump(vocab_id, 'data/LJSpeech/vocabulary_id.pkl')
