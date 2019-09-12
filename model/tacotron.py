import tensorflow as tf

from model.networks import encoder, decoder
from constants.hparams import Hyperparams as hparams
from utils.logger import log


class Tacotron():
  """ A Complete Tacotron Model """
  def __init__(self):
    pass


  def init(self, inputs, input_lengths, mel_targets=None, linear_targets=None):
    """ Initialize the model for inference
    Sets "mel_outputs", "linear_outputs", and "alignments" fields.
    
    @param  inputs          int32 Tensor with shape [N, T_in] where N is batch size, T_in is number of
                            steps in the input time series, and values are character IDs
    @param  input_lengths:  int32 Tensor with shape [N] where N is batch size and values are the lengths
                            of each sequence in inputs.
    @param  mel_targets     float32 Tensor with shape [N, T_out, M] where N is batch size, T_out is number
                            of steps in the output time series, M is num_mels, and values are entries in the mel
                            spectrogram. Only needed for training.
    @param  linear_targets  float32 Tensor with shape [N, T_out, F] where N is batch_size, T_out is number
                            of steps in the output time series, F is num_freq, and values are entries in the linear
                            spectrogram. Only needed for training.
    """

    with tf.variable_scope('inference') as scope:
      is_training = linear_targets is not None
      batch_size = tf.shape(inputs)[0]

      log('----------------------------------------------------------------')
      log('Initialized Tacotron model with dimensions: ')
      
      # encoder
      encoder_outputs = encoder(inputs, input_lengths, is_training)

      # decoder
      linear_outputs, mel_outputs, alignments = decoder(inputs, encoder_outputs, is_training, batch_size, mel_targets)

      self.inputs = inputs
      self.input_lengths = input_lengths
      self.mel_outputs = mel_outputs
      self.linear_outputs = linear_outputs
      self.alignments = alignments
      self.mel_targets = mel_targets
      self.linear_targets = linear_targets

      log('----------------------------------------------------------------')


  def add_loss(self):
    """ Adding Loss to the model """
    with tf.variable_scope('loss'):
      self.mel_loss = tf.reduce_mean(tf.abs(self.mel_targets - self.mel_outputs))
      l1_loss = tf.abs(self.linear_targets - self.linear_outputs)
      
      # prioritize loss for freqeuncies under 3000 Hz
      n_priority_freq = int(3000 / (hparams.sample_rate * 0.5) * hparams.num_freq)

      self.linear_loss = 0.5 * tf.reduce_mean(l1_loss) + 0.5 * tf.reduce_mean(l1_loss[:,:,0:n_priority_freq])
      self.loss = self.mel_loss + self.linear_loss


  def add_optimizer(self, global_step):
    """ Adding optimizer to the model """
    with tf.variable_scope('optimizer'):
      if (hparams.learning_rate_decay):
        self.learning_rate = _learning_rate_decay(hparams.initial_lr, global_step)
      else:
        self.learning_rate = tf.convert_to_tensor(hparams.initial_lr)

      optimizer = tf.train.AdamOptimizer(self.learning_rate, hparams.adam_beta_1, hparams.adam_beta_2)
      gradients, variables = zip(*optimizer.compute_gradients(self.loss))
      
      self.gradients = gradients
      clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

      with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
          global_step=global_step)


def _learning_rate_decay(initial_lr, global_step):
  """ Learning rate decay 
  
  @param  initial_lr    initial learning rate
  @param  global_step   global step number
  """

  warmup_step = 4000.0
  step = tf.cast(global_step + 1, dtype=tf.float32)

  return initial_lr * warmup_step**0.5 * tf.minimum(step * warmup_step**-1.5, step**-0.5)
