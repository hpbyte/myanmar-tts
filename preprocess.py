import os
import argparse
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from tqdm import tqdm

import constants.hparams as hparams
from signal_proc import audio


def prepare_audio_dataset(in_dir, out_dir, nb_workers=1, tqdm=lambda x: x):
  """
  Preprocess the dataset of the audio files from a given input path into a given output path

  @type   in_dir      str
  @type   out_dir     str
  @type   nb_workers  int
  @type   tqdm        lambda

  @param  in_dir      directory which contains speech corpus
  @param  out_dir     directory in which the training data will be created
  @param  nb_workers  number of parallel processes
  @param  tqdm        for progress bar

  @rtype              list
  @return             a list of tuples describing the training examples
  """

  executor = ProcessPoolExecutor(max_workers=nb_workers)
  futures = []
  indx = 1
  with open(os.path.join(in_dir, 'metadata.csv'), encoding='utf-8') as f:
    for line in f:
      parts = line.strip().split(',')
      txt = parts[0]
      wav_path = os.path.join(in_dir, 'wavs', '%s' % parts[1])
      futures.append(executor.submit(partial(process_utterance, out_dir, indx, wav_path, txt)))
      indx += 1

  return [future.result() for future in tqdm(futures)]


def prepare_text_dataset(metadata, out_dir):
  """
  Preprocess the dataset of the texts from a given input path into a given output path.
  This writes a file called train.txt as the input dataset

  @type   metadata    list
  @type   out_dir     str

  @param  metadata    text data
  @param  out_dir     output directory for the preprocessed texts
  """
  with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
    for m in metadata:
      f.write('|'.join([str(x) for x in m]) + '\n')


def process_utterance(out_dir, index, wav_path, text):
  """
  Preprocess a single <text, audio> pair and outputs both linear and mel spectrograms and a tuple about them.

  @type   out_dir   str
  @type   index     int
  @type   wav_path  str
  @type   text      str

  @param  out_dir   output directory for spectrograms
  @param  index     index for spectrogram filenames
  @param  wav_path  path to the audio file
  @param  text      the text spoken in the input audio

  @rtype            tuple
  @return           a (spectrogram_filename, mel_filename, n_frames, text) tuple for train.txt
  """

  # load the audio to a numpy array
  wav = audio.load_audio(wav_path)

  # compute linear-scale spectrogram
  spectrogram = audio.wav_to_spectrogram(wav).astype(np.float32)
  n_frames = spectrogram.shape[1]

  # compute mel-scale spectrogram
  mel_spectrogram = audio.wav_to_melspectrogram(wav).astype(np.float32)

  # outputs the spectrograms
  spectrogram_filename = 'mmspeech-spec-%05d.npy' % index
  mel_spectrogram_filename = 'mmspeech-mel-%05d.npy' % index
  np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
  np.save(os.path.join(out_dir, mel_spectrogram_filename), mel_spectrogram.T, allow_pickle=False)

  # return the tuple
  return (spectrogram_filename, mel_spectrogram_filename, n_frames, text)


def preprocess(args):
  in_dir = os.path.join(args.base_dir, args.input)
  out_dir = os.path.join(args.base_dir, args.output)
  os.makedirs(out_dir, exist_ok=True)
  metadata = prepare_audio_dataset(in_dir, out_dir, args.nb_workers, tqdm=tqdm)
  prepare_text_dataset(metadata, out_dir)
  # give feedback
  frames = sum([m[2] for m in metadata])
  hours = frames * hparams.FRAME_SHIFT / 3600
  print('Wrote %d utterances, %d frames (%.2f hours)' % (len(metadata), frames, hours))
  print('Max input length: %d' % max(len(m[3]) for m in metadata))
  print('Max output length %d' % max(m[2] for m in metadata))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', default=os.path.expanduser('~/Documents/myanmar-tts'))
  parser.add_argument('--input', default='mmSpeech')
  parser.add_argument('--output', default='training')
  parser.add_argument('--nb_workers', type=int, default=cpu_count())
  args = parser.parse_args()

  preprocess(args)


if __name__ == "__main__":
  main()
