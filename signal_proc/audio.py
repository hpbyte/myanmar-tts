import math

import tensorflow as tf
import scipy
import numpy as np
import librosa

import constants.hparams as hparams


# Utils
def load_audio(path):
  return librosa.core.load(path, sr=hparams.SAMPLE_RATE)[0]


def save_audio(wav, path):
  wav *= 32767 / max(0.01, np.max(np.abs(wav)))
  scipy.io.wavfile.write(path, hparams.SAMPLE_RATE, wav.astype(np.int16))


# Unit Conversions
def amp_to_db(x):
  return 20 * np.log10(np.maximum(1e-5, x))


def db_to_amp(x):
  return np.power(10.0, x * 0.05)


def db_to_amp_tf(x):
  return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)


def normalize(S):
  return np.clip((S - hparams.MIN_LEVEL_DB) / -hparams.MIN_LEVEL_DB, 0, 1)


def denormalize(S):
  return (np.clip(S, 0, 1) * -hparams.MIN_LEVEL_DB) + hparams.MIN_LEVEL_DB


def denormalize_tf(S):
  return (tf.clip_by_value(S, 0, 1) * -hparams.MIN_LEVEL_DB) + hparams.MIN_LEVEL_DB


# Signal Processing formulas
def stft(y):
  """ Short-Time-Fourier-Transform """
  return librosa.stft(y=y, n_fft=hparams.N_FFT, hop_length=hparams.HOP_LENGTH, win_length=hparams.WIN_LENGTH)


def stft_tf(signals):
  """ Short-Time-Fourier-Transform in TensorFlow """
  return tf.contrib.signal.stft(signals, hparams.WIN_LENGTH, hparams.HOP_LENGTH, hparams.N_FFT, pad_end=False)


def inv_stft(y):
  """ Inverse-Short-Time-Fourier-Transform """
  return librosa.istft(y, hop_length=hparams.HOP_LENGTH, win_length=hparams.WIN_LENGTH)


def inv_stft_tf(stfts):
  """ Inverse-Short-Time-Fourier-Transform in TensorFlow """
  return tf.contrib.signal.inverse_stft(stfts, hparams.WIN_LENGTH, hparams.HOP_LENGTH, hparams.N_FFT)


def preemphasis(x):
  return scipy.signal.lfilter([1, -hparams.PREEMPHASIS], [1], x)


def inv_preemphasis(x):
  return scipy.signal.lfilter([1], [1, -hparams.PREEMPHASIS], x)


# Linear-scale and Mel-scale Spectrograms
def wav_to_spectrogram(y):
  """ waveform to spectrogram conversion """
  spectro = np.abs(stft(preemphasis(y)))
  S = amp_to_db(spectro) - hparams.REF_LEVEL_DB
  return normalize(S)


def spectrogram_to_wav(S):
  """ spectrogram to waveform conversion """
  spectro = db_to_amp(denormalize(S) + hparams.REF_LEVEL_DB)
  return inv_preemphasis(Griffin_Lim(spectro ** hparams.POWER))


def spectrogram_to_wav_tf(S):
  """ spectrogram to waveform conversion in TensorFlow (without inv_preemphasis) """
  spectro = db_to_amp_tf(denormalize_tf(S) + hparams.REF_LEVEL_DB)
  return Griffin_Lim_tf(tf.pow(spectro, hparams.POWER))


def wav_to_melspectrogram(y):
  """ waveform to mel-scale spectrogram conversion """
  mel_transform_matrix = librosa.filters.mel(hparams.SAMPLE_RATE, hparams.N_FFT, n_mels=hparams.NUM_MELS)
  spectro = np.abs(stft(preemphasis(y)))
  mel_spectro = np.dot(mel_transform_matrix, spectro)
  S = amp_to_db(mel_spectro) - hparams.REF_LEVEL_DB
  return normalize(S)


# Griffin-Lim Reconstruction Algorithm
def Griffin_Lim(S):
  angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
  S_complex = np.abs(S).astype(np.complex)
  y = inv_stft(S_complex * angles)

  for i in range(hparams.GRIFFIN_LIM_ITERS):
    angles = np.exp(1j * np.angle(stft(y)))
    y = inv_stft(S_complex * angles)

  return y


def Griffin_Lim_tf(S):
  with tf.variable_scope('griffinlim'):
    S = tf.expand_dims(S, 0)
    S_complex = tf.identity(tf.cast(S, dtype=tf.complex64))
    y = inv_stft_tf(S_complex)
    
    for i in range(hparams.GRIFFIN_LIM_ITERS):
      est = stft_tf(y)
      angles = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)
      y = inv_stft_tf(S_complex * angles)

    return tf.squeeze(y, 0)


def find_endpoint(wav, threshold_db = -40, min_silence_sec = 0.8):
  window_length = int(hparams.SAMPLE_RATE * min_silence_sec)
  hop_length = int(window_length / 4)
  threshold = db_to_amp(threshold_db)

  for x in range(hop_length, len(wav) - window_length, hop_length):
    if np.max(wav[x:(x + window_length)]) < threshold:
      return x + hop_length

  return len(wav)
