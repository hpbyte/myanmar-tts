import numpy as np
from tqdm import tqdm

def transform_text(list_of_strings, vocab_ids, max_length):
  transformed_data = []

  # create a progress bar with tqdm
  for str in tqdm(list_of_strings):
    list_of_char = list(str)
    list_of_char_id = [vocab_ids[char] for char in list_of_char]

    # number of characters
    nb_char = len(list_of_char_id)

    # padding for fixed length input
    if nb_char < max_length:
      for i in range(max_length - nb_char):
        list_of_char_id.append(vocab_ids['P'])

    transformed_data.append(list_of_char_id)

  training_input = np.array(transformed_data)

  return training_input
