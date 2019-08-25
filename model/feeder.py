import os
import random
import time
import traceback
import threading

import numpy as np
import tensorflow as tf

from constants.hparams import Hyperparams as hparams
from text.tokenizer import text_to_sequence
from utils.logger import log

batches_per_group = 32
pad_val = 0

class DataFeeder(threading.Thread):
  """ Feeds batches of data on a queue in a background thread """

  def __init__(self, coordinator, metadata_filename):
    super(DataFeeder, self).__init__()
    self._coordi = coordinator
    self._hparams = hparams
    self._offset = 0

    # load metadata
    self._data_dir = os.path.dirname(metadata_filename)
    with open(metadata_filename, encoding='utf-8') as f:
      self._metadata = [line.strip().split('|') for line in f]
      hours = sum((int(x[2]) for x in self._metadata)) * hparams.frame_shift / 3600
      log('Loaded metadata for %d examples (%.2f hours)' % (len(self._metadata), hours))

    # create placeholders for inputs and targets
    # didn't specify batch size bcuz of the need to feed different sized batches at eval time
    self._placeholders = [
      tf.compat.v1.placeholder(tf.int32, [None, None], 'inputs'),
      tf.compat.v1.placeholder(tf.int32, [None], 'input_lengths'),
      tf.compat.v1.placeholder(tf.float32, [None, None, hparams.num_mels], 'mel_targets'),
      tf.compat.v1.placeholder(tf.float32, [None, None, hparams.num_freq], 'linear_targets')
    ]

    # create a queue for buffering data
    queue = tf.FIFOQueue(8, [tf.int32, tf.int32, tf.float32, tf.float32], name='input_queue')
    self._enqueue_op = queue.enqueue(self._placeholders)
    self.inputs, self.input_lengths, self.mel_targets, self.linear_targets = queue.dequeue()
    self.inputs.set_shape(self._placeholders[0].shape)
    self.input_lengths.set_shape(self._placeholders[1].shape)
    self.mel_targets.set_shape(self._placeholders[2].shape)
    self.linear_targets.set_shape(self._placeholders[3].shape)


  def start_in_session(self, session):
    self._session = session
    # starting the thread which in turn, invokes the run() method
    self.start()


  def run(self):
    # perform queueing operations until it should stop
    try:
      while not self._coordi.should_stop():
        # if it shoudn't stop, enqueue the next batches
        self.enqueue_next_group()
    except Exception as e:
      # print the exception occurred
      traceback.print_exc()
      # tell the coordinator to stop
      self._coordi.request_stop(e)


  def enqueue_next_group(self):
    """ Enqueue next group of batches into the queue """

    start = time.time()

    # read a group of examples
    nb_batches = self._hparams.batch_size
    r = self._hparams.outputs_per_step
    examples = [self.get_next_example() for i in range(nb_batches * batches_per_group)]

    # sort examples based on their length for efficiency
    examples.sort(key=lambda x: x[-1])
    batches = [examples[i:i+nb_batches] for i in range(0, len(examples), nb_batches)]
    random.shuffle(batches)

    log('Generated %d batches of size %d in %0.3f sec' % (len(batches), nb_batches, time.time() - start))

    for b in batches:
      # make a feeding dictionary of iterables with the placeholders mapping to the input data
      # {
      #   (inputs => input_data_text), (input_lengths => input_data_lengths),
      #   (mel_targets => input_mel_targets), (linear_targets => input_linear_targets)
      # }
      feed_dict = dict(zip(self._placeholders, _prepare_batch(b, r)))
      # run the session with the fed placeholders
      self._session.run(self._enqueue_op, feed_dict=feed_dict)


  def get_next_example(self):
    """
    Get a single example (input, mel_target, linear_target, cost) from metadata.
    This read the metadata file by offsetting the position.
    """

    if self._offset >= len(self._metadata):
      # if somehow the offset gets larger than metadata size,
      # set the offset back to 0 and shuffle the metadata
      self._offset = 0
      random.shuffle(self._metadata)

    meta = self._metadata[self._offset]
    self._offset += 1

    text = meta[3]
    # get the normalized sequence of text
    input_data = np.asarray(text_to_sequence(text), dtype=np.int32)
    # load the  linear spectrogram.npy
    linear_target = np.load(os.path.join(self._data_dir, meta[0]))
    # laod the mel-spectrogram.npy
    mel_target = np.load(os.path.join(self._data_dir, meta[1]))

    return (input_data, mel_target, linear_target, len(linear_target))


# helper functions
def _prepare_batch(batch, outputs_per_step):
  """
  Having constant input length is essential for training,
  so we need to pad each of them if needed
  """
  random.shuffle(batch)
  # since a single example looks like this (input, mel_target, linear_target, cost),
  # x[0] => inputs, x[1] => mel_target, x[2] => linear_target
  inputs = _get_padded_inputs([x[0] for x in batch])
  input_lengths = np.asarray([len(x[0]) for x in batch], dtype=np.int32)
  
  mel_targets = _get_padded_targets([x[1] for x in batch], outputs_per_step)
  linear_targets = _get_padded_targets([x[2] for x in batch], outputs_per_step)

  return (inputs, input_lengths, mel_targets, linear_targets)


def _get_padded_inputs(inputs):
  """ join a sequence of arrays of padded inputs """
  max_len = max((len(x) for x in inputs))
  return np.stack([_pad_input(x, max_len) for x in inputs])


def _get_padded_targets(targets, alignment):
  """ join a sequence of arrays of padded targets """
  max_len = max((len(t) for t in targets)) + 1
  # make the max_len to be a multiple of outputs_per_step 
  return np.stack([_pad_target(t, _round_up(max_len, alignment)) for t in targets])


def _pad_input(x, max_len):
  """ pad the input (whose length is lower than the max_len) with zeros """
  return np.pad(x, (0, max_len - x.shape[0]), mode='constant', constant_values=pad_val)


def _pad_target(t, max_len):
  """ pad the target (whose length is lower than the max_len) with zeros """
  return np.pad(t, [(0, max_len - t.shape[0]), (0, 0)], mode='constant', constant_values=pad_val)


def _round_up(x, multiple):
  """ get the rounded max_len for a target """
  remainder = x % multiple
  return x if remainder == 0 else x + multiple - remainder
