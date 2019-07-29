import numpy as np
import pandas as pd
from scipy import signal, io
import librosa
import os
import copy
from constants import *

def normalize(S):
  return np.clip((S - REF_DB + MAX_DB) / MAX_DB, 1e-8, 1)


def denormalize(S):
  return ((np.clip(S, 0, 1) * MAX_DB) - MAX_DB + REF_DB)


def amp_to_db(S):
  return 20 * np.log10(np.maximum(1e-5, S))


def db_to_amp(S):
  return np.power(10.0, S * 0.05)


def transpose(S):
  return S.T.astype(np.float32)


def preemphasis(x):
  # use pre-emphasis to filter out low frequencies
  return signal.lfilter([1, -PREEMPHASIS], [1], x)


def inv_preemphasis(x):
  return signal.lfilter([1], [1, -PREEMPHASIS], x)


def linear_to_mel(S):
  mel_transform_matrix = librosa.filters.mel(SAMPLING_RATE, N_FFT, N_MEL)
  return np.dot(mel_transform_matrix, S)


def mel_spectrogram(x):
  mel_spectro = stft(preemphasis(x))
  mel_spectro = amp_to_db(linear_to_mel(np.abs(mel_spectro)))
  mel_spectro = transpose(normalize(mel_spectro))

  return mel_spectro


def spectrogram(x):
  stft_matrix = stft(preemphasis(x))
  spectro = amp_to_db(np.abs(stft_matrix))
  spectro = transpose(normalize(spectro))

  return spectro


def stft(y):
  return librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window=WINDOW_TYPE)


def istft(y):
  return librosa.istft(y, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window=WINDOW_TYPE)


def get_spectros(file_path):
  waveform, _ = librosa.load(file_path, sr=SAMPLING_RATE)

  waveform, _ = librosa.effects.trim(waveform)

  # compute magnitutde and mel-spectrogram
  spectro = spectrogram(waveform)
  mel_spectro = mel_spectrogram(waveform)

  return mel_spectro, spectro


def get_padded_spectros(file_path):
  """
  pad the spectrogram if its total length is not a multiple of r, so that it becomes a multiple of r
  """
  file_name = os.path.basename(file_path)
  mel_spectro, spectro = get_spectros(file_path)
  t = mel_spectro.shape[0]

  # no. of paddings for reduction
  nb_paddings = r - (t % r) if t % r != 0 else 0

  mel_spectro = np.pad(mel_spectro, [[0, nb_paddings], [0, 0]], mode='constant')
  spectro = np.pad(spectro, [[0, nb_paddings], [0, 0]], mode='constant')

  return file_name, mel_spectro.reshape((-1, N_MEL * r)), spectro


def griffin_lim(S):
  """
  Griffin-Lim Reconstruction Algorithm
  """
  spectro = copy.deepcopy(S)

  for i in range(N_ITER):
    estimated_wav = istft(spectro)
    estimated_stft = stft(estimated_wav)

    phase = estimated_stft / np.maximum(1e-8, np.abs(estimated_stft))
    spectro = S * phase

  estimated_wav = istft(spectro)

  return np.real(estimated_wav)


def spectro_to_wav(S):
  """
  Converts spectrogram to corresponding waveform
  """
  W = db_to_amp(denormalize(S.T))       # convert back to linear
  W = inv_preemphasis(griffin_lim(W)) # reconstruct phase
  
  waveform, _ = librosa.effects.trim(W)

  return waveform.astype(np.float32)
  