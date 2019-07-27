import pandas as pd
import numpy as np
from sklearn.externals import joblib
from tqdm import tqdm
from processing.proc_audio import get_padded_spectros
from constants import *
import tensorflow as tf

sess = tf.Session()

print('Loading data ...')
metadata = pd.read_csv(BASE_DIR + 'LJSpeech-1.1/metadata.csv', dtype='object', quoting=3, sep='|', header=None)

metadata = metadata.iloc[:500]

# audio filenames
wav_filenames = metadata[0].values

mel_spectro_data = []
spectro_data = []
decoder_input = []

print('Processing the audio samples (computation of spectrograms) ...')
for file_name in tqdm(wav_filenames):
  file_path = BASE_DIR + 'LJSpeech-1.1/wavs/' + file_name + '.wav'
  fname, mel_spectro, spectro = get_padded_spectros(file_path, r, PREEMPHASIS, N_FFT, HOP_LENGTH, WIN_LENGTH, SAMPLING_RATE, N_MEL, REF_DB, MAX_DB)

  decod_input_tensor = tf.concat((tf.zeros_like(mel_spectro[:-1, :]), mel_spectro[:-1, :]), 0)

  decod_input = sess.run(decod_input_tensor)
  decod_input = decod_input[:, -N_MEL:]

  # padding of the temporal dimension
  dim0_mel_spectro = mel_spectro.shape[0]
  dim1_mel_spectro = mel_spectro.shape[1]
  padded_mel_spectro = np.zeros((MAX_MEL_TIME_LENGTH, dim1_mel_spectro))
  padded_mel_spectro[:dim0_mel_spectro, :dim1_mel_spectro] = mel_spectro

  dim0_spectro = spectro.shape[0]
  dim1_spectro = spectro.shape[1]
  padded_spectro = np.zeros((MAX_MAG_TIME_LENGTH, dim1_spectro))
  padded_spectro[:dim0_spectro, :dim1_spectro] = spectro

  dim0_decode_input = decod_input.shape[0]
  dim1_decode_input = decod_input.shape[1]
  padded_decod_input = np.zeros((MAX_MEL_TIME_LENGTH, dim1_decode_input))
  padded_decod_input[:dim0_decode_input, :dim1_decode_input] = decod_input

  mel_spectro_data.append(padded_mel_spectro)
  spectro_data.append(padded_spectro)
  decod_input.append(padded_decod_input)

print('Convert into np.array')
mel_spectro_data_array = np.array(mel_spectro_data)
spectro_data_array = np.array(spectro_data)
decoder_input_array = np.array(decod_input)

print('Split into training and testing')
len_train = int(TRAIN_SET_RATIO * len(metadata))

mel_spectro_data_array_training = mel_spectro_data_array[:len_train]
mel_spectro_data_array_testing = mel_spectro_data_array[len_train:]

spectro_data_array_training = spectro_data_array[:len_train]
spectro_data_array_testing = spectro_data_array[len_train:]

decoder_input_array_training = decoder_input_array[:len_train]
decoder_input_array_testing = decoder_input_array[len_train:]

print('Save data as plk')
joblib.dump(mel_spectro_data_array_training, 'data/LJSpeech/mel_spectro_training.plk')
joblib.dump(mel_spectro_data_array_testing, 'data/LJSpeech/mel_spectro_testing.plk')

joblib.dump(spectro_data_array_training, 'data/LJSpeech/spectro_training.plk')
joblib.dump(spectro_data_array_testing, 'data/LJSpeech/spectro_testing.plk')

joblib.dump(decoder_input_array_training, 'data/LJSpeech/decoder_input_training.plk')
joblib.dump(decoder_input_array_testing, 'data/LJSpeech/decoder_input_testing.plk')
