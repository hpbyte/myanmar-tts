import tensorflow as tf

from model.modules import (prenet, encoder_cbhg, post_cbhg, attention_decoder,
                          Decoder_Prenet, ConcatOutputAndAttentionWrapper)
from model.helpers import (TacoTrainingHelper, TacoTestHelper)
import constants.hparams as hparams
from text.character_set import characters
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

    with tf.variable_scope('inference'):
      is_training = linear_targets is not None
      batch_size = tf.shape(inputs)[0]

      if (is_training):
        helper = TacoTrainingHelper(inputs, mel_targets, hparams.NUM_MELS, hparams.OUTPUTS_PER_STEP)
      else:
        helper = TacoTestHelper(batch_size, hparams.NUM_MELS, hparams.OUTPUTS_PER_STEP)

      """ Character Embeddings """
      embedding_table = tf.get_variable('embedding', [len(characters), 256],
                          initializer=tf.truncated_normal_initializer(stddev=0.5))

      embedded_inputs = tf.nn.embedding_lookup(embedding_table, inputs)       # [N, T_in, embed_depth=256]

      """ Encoder """
      prenet_outputs = prenet(embedded_inputs, is_training)                   # [N, T_in, prenet_depth=128]
      encoder_outputs = encoder_cbhg(prenet_outputs, is_training)             # [N, T_in, prenet_depth=256]

      """ Decoder 1 """
      attention_cell = attention_decoder(encoder_outputs)                     # [N, T_in, attention_depth=256]

      # apply prenet before concatanation in AttentionWrapper
      attention_cell = Decoder_Prenet(attention_cell, is_training, [256, 128])

      # Concatenate attention context vector and RNN cell output into a 2*attention_depth=512D vector.
      concat_cell = ConcatOutputAndAttentionWrapper(attention_cell)           # [N, T_in, 2*attention_depth=512]

      decoder_cell = tf.keras.layers.StackedRNNCells([
        tf.contrib.rnn.OutputProjectionWrapper(concat_cell, 256),
        tf.nn.rnn_cell.ResidualWrapper(tf.keras.layers.GRUCell(256)),
        tf.nn.rnn_cell.ResidualWrapper(tf.keras.layers.GRUCell(256))
      ])                                                                      # [N, T_in, decoder_depth=256]

      # project onto r mel_spectrograms (predict r outputs at each RNN step)
      output_cell = tf.contrib.rnn.OutputProjectionWrapper(decoder_cell, hparams.NUM_MELS * hparams.OUTPUTS_PER_STEP)
      decoder_init_state = output_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

      """ Decoder 2 """
      (decoder_outputs, _), decoder_final_state, _ = tf.contrib.seq2seq.dynamic_decode(
                                                      tf.contrib.seq2seq.BasicDecoder(output_cell, helper, decoder_init_state),
                                                      maximum_iterations=hparams.MAX_ITERS
                                                    )                         # [N, T_out/r, M*r]

      # reshape outputs to be one output per entry
      mel_outputs = tf.reshape(decoder_outputs, [batch_size, -1, hparams.NUM_MELS]) # [N, T_out, M]

      # post-processing CBHG
      post_outputs = post_cbhg(mel_outputs, hparams.NUM_MELS, is_training)    # [N, T_out, postnet_depth=256]

      linear_outputs = tf.keras.layers.Dense(hparams.NUM_FREQ)(post_outputs)  # [N, T_out, F]

      # grab alignments from the final state
      alignments = tf.transpose(decoder_final_state[0].alignment_history.stack(), [1, 2, 0])

      self.inputs = inputs
      self.input_lengths = input_lengths
      self.mel_outputs = mel_outputs
      self.linear_outputs = linear_outputs
      self.alignments = alignments
      self.mel_targets = mel_targets
      self.linear_targets = linear_targets
      log('Initialized Tacotron model with dimensions: ')
      log('  embedding:                 %d' % embedded_inputs.shape[-1])
      log('  prenet out:                %d' % prenet_outputs.shape[-1])
      log('  encoder out:               %d' % encoder_outputs.shape[-1])
      log('  attention out:             %d' % attention_cell.output_size)
      log('  concat attn & out:         %d' % concat_cell.output_size)
      log('  decoder cell out:          %d' % decoder_cell.output_size)
      log('  decoder out (%d frames):   %d' % (hparams.OUTPUTS_PER_STEP, decoder_outputs.shape[-1]))
      log('  decoder out (1 frame):     %d' % mel_outputs.shape[-1])
      log('  postnet out:               %d' % post_outputs.shape[-1])
      log('  linear out:                %d' % linear_outputs.shape[-1])


  def add_loss(self):
    """ Adding Loss to the model """
    with tf.variable_scope('loss'):
      self.mel_loss = tf.reduce_mean(tf.abs(self.mel_targets - self.mel_outputs))
      l1_loss = tf.abs(self.linear_targets - self.linear_outputs)
      
      # prioritize loss for freqeuncies under 3000 Hz
      n_priority_freq = int(3000 / (hparams.SAMPLE_RATE * 0.5) * hparams.NUM_FREQ)
      
      self.linear_loss = 0.5 * tf.reduce_mean(l1_loss) + 0.5 * tf.reduce_mean(l1_loss[:,:,0:n_priority_freq])
      self.loss = self.mel_loss + self.linear_loss


  def add_optimizer(self, global_step):
    """ Adding optimizer to the model """
    with tf.variable_scope('optimizer'):
      if (hparams.LEARNING_RATE_DECAY):
        self.learning_rate = _learning_rate_decay(hparams.INITIAL_LR, global_step)
      else:
        self.learning_rate = tf.convert_to_tensor(hparams.INITIAL_LR)

      optimizer = tf.train.AdamOptimizer(self.learning_rate, hparams.ADAM_BETA_1, hparams.ADAM_BETA_2)
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

  return initial_lr * warmup_step**0.5 * tf.minimum(step * warmup_step**-1.5 * step**-0.5)
