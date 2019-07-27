import numpy as np
import pandas as pd
import librosa
import os

def normalize(S, ref_db, max_db):
  return np.clip((S - ref_db + max_db) / max_db, 1e-8, 1)

def to_decibel(S):
  return 20 * np.log10(np.maximum(1e-5, S))

def transpose(S):
  return S.T.astype(np.float32)

def get_spectros(file_path, preemphasis, n_fft, hop_length, win_length, sampling_rate, n_mel, ref_db, max_db):
  waveform, sampling_rate = librosa.load(file_path, sr=sampling_rate)

  waveform, _ = librosa.effects.trim(waveform)

  # use pre-emphasis to filter out low frequencies
  waveform = np.append(waveform[0], waveform[1:] - preemphasis * waveform[:-1])

  # compute Short Time Fourier Transform
  stft_matrix = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

  # compute magnitutde and mel-spectrogram
  spectro = np.abs(stft_matrix)

  mel_transform_matrix = librosa.filters.mel(sampling_rate, n_fft, n_mel, htk=True)

  mel_spectro = np.dot(mel_transform_matrix, spectro)

  # use decibel scale
  mel_spectro = to_decibel(mel_spectro)
  spectro = to_decibel(spectro)

  # normalize the spectrograms
  mel_spectro = normalize(mel_spectro, ref_db, max_db)
  spectro = normalize(spectro, ref_db, max_db)

  # transpose the spectrograms to have the time as 1st dimension and frequency as 2nd dimension
  mel_spectro = transpose(mel_spectro)
  spectro = transpose(spectro)

  return mel_spectro, spectro


def get_padded_spectros(file_path, r, preemphasis, n_fft, hop_length, win_length, sampling_rate, n_mel, ref_db, max_db):
  """
  pad the spectrogram if its total length is not a multiple of r, so that it becomes a multiple of r
  """
  file_name = os.path.basename(file_path)
  mel_spectro, spectro = get_spectros(file_path, preemphasis, n_fft, hop_length, win_length, sampling_rate, n_mel, ref_db, max_db)
  t = mel_spectro.shape[0]

  # no. of paddings for reduction
  nb_paddings = r - (t % r) if t % r != 0 else 0

  mel_spectro = np.pad(mel_spectro, [[0, nb_paddings], [0, 0]], mode='constant')

  spectro = np.pad(spectro, [[0, nb_paddings], [0, 0]], mode='constant')

  return file_name, mel_spectro.reshape((-1, n_mel * r)), spectro
