import math

import tensorflow as tf
import scipy
import numpy as np
import librosa

from constants.hparams import Hyperparams as hparams


# Utils
def load_audio(path):
  return librosa.core.load(path, sr=hparams.sample_rate)[0]


def save_audio(wav, path):
  wav *= 32767 / max(0.01, np.max(np.abs(wav)))
  scipy.io.wavfile.write(path, hparams.sample_rate, wav.astype(np.int16))


# Unit Conversions
def amp_to_db(x):
  return 20 * np.log10(np.maximum(1e-5, x))


def db_to_amp(x):
  return np.power(10.0, x * 0.05)


def db_to_amp_tf(x):
  return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)


def normalize(S):
  return np.clip((S - hparams.min_db) / -hparams.min_db, 0, 1)


def denormalize(S):
  return (np.clip(S, 0, 1) * -hparams.min_db) + hparams.min_db


def denormalize_tf(S):
  return (tf.clip_by_value(S, 0, 1) * -hparams.min_db) + hparams.min_db


# Signal Processing formulas
def stft(y):
  """ Short-Time-Fourier-Transform """
  return librosa.stft(y=y, n_fft=hparams.n_fft, hop_length=hparams.hop_length, win_length=hparams.win_length)


def stft_tf(signals):
  """ Short-Time-Fourier-Transform in TensorFlow """
  return tf.contrib.signal.stft(signals, hparams.win_length, hparams.hop_length, hparams.n_fft, pad_end=False)


def inv_stft(y):
  """ Inverse-Short-Time-Fourier-Transform """
  return librosa.istft(y, hop_length=hparams.hop_length, win_length=hparams.win_length)


def inv_stft_tf(stfts):
  """ Inverse-Short-Time-Fourier-Transform in TensorFlow """
  return tf.contrib.signal.inverse_stft(stfts, hparams.win_length, hparams.hop_length, hparams.n_fft)


def preemphasis(x):
  return scipy.signal.lfilter([1, -hparams.preemphasis], [1], x)


def inv_preemphasis(x):
  return scipy.signal.lfilter([1], [1, -hparams.preemphasis], x)


# Linear-scale and Mel-scale Spectrograms
def wav_to_spectrogram(y):
  """ waveform to spectrogram conversion """
  spectro = np.abs(stft(preemphasis(y)))
  S = amp_to_db(spectro) - hparams.ref_db
  return normalize(S)


def spectrogram_to_wav(S):
  """ spectrogram to waveform conversion """
  spectro = db_to_amp(denormalize(S) + hparams.ref_db)
  return inv_preemphasis(Griffin_Lim(spectro ** hparams.power))


def spectrogram_to_wav_tf(S):
  """ spectrogram to waveform conversion in TensorFlow (without inv_preemphasis) """
  spectro = db_to_amp_tf(denormalize_tf(S) + hparams.ref_db)
  return Griffin_Lim_tf(tf.pow(spectro, hparams.power))


def wav_to_melspectrogram(y):
  """ waveform to mel-scale spectrogram conversion """
  mel_transform_matrix = librosa.filters.mel(hparams.sample_rate, hparams.n_fft, n_mels=hparams.num_mels)
  spectro = np.abs(stft(preemphasis(y)))
  mel_spectro = np.dot(mel_transform_matrix, spectro)
  S = amp_to_db(mel_spectro) - hparams.ref_db
  return normalize(S)


# Griffin-Lim Reconstruction Algorithm
def Griffin_Lim(S):
  angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
  S_complex = np.abs(S).astype(np.complex)
  y = inv_stft(S_complex * angles)

  for i in range(hparams.griffin_lim_iters):
    angles = np.exp(1j * np.angle(stft(y)))
    y = inv_stft(S_complex * angles)

  return y


def Griffin_Lim_tf(S):
  with tf.compat.v1.variable_scope('griffinlim'):
    S = tf.expand_dims(S, 0)
    S_complex = tf.identity(tf.cast(S, dtype=tf.complex64))
    y = inv_stft_tf(S_complex)
    
    for i in range(hparams.griffin_lim_iters):
      est = stft_tf(y)
      angles = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)
      y = inv_stft_tf(S_complex * angles)

    return tf.squeeze(y, 0)


def find_endpoint(wav, threshold_db = -40, min_silence_sec = 0.8):
  window_length = int(hparams.sample_rate * min_silence_sec)
  hop_length = int(window_length / 4)
  threshold = db_to_amp(threshold_db)

  for x in range(hop_length, len(wav) - window_length, hop_length):
    if np.max(wav[x:(x + window_length)]) < threshold:
      return x + hop_length

  return len(wav)
