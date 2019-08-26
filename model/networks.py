import tensorflow as tf

from model.modules import (prenet, encoder_cbhg, post_cbhg, attention_decoder,
                          Decoder_Prenet, ConcatOutputAndAttentionWrapper)
from model.helpers import (TacoTrainingHelper, TacoTestHelper)
from constants.hparams import Hyperparams as hparams
from text.character_set import characters
from utils.logger import log


def encoder(inputs, is_training):
  """ Encoder 
  
  Embeddings -> Prenet -> Encoder CBHG

  @param    inputs        int32 Tensor with shape [N, T_in] where N is batch size, T_in is number of
                          steps in the input time series, and values are character IDs
  @param    is_training   flag for training or eval

  @returns                outputs from the encoder
  """

  # Character Embeddings
  embedding_table = tf.compat.v1.get_variable('embedding', [len(characters), hparams.embed_depth],
                      initializer=tf.truncated_normal_initializer(stddev=0.5))

  embedded_inputs = tf.nn.embedding_lookup(embedding_table, inputs)       # [N, T_in, embed_depth=256]

  prenet_outputs = prenet(embedded_inputs, is_training)                   # [N, T_in, prenet_depth=128]
  
  encoder_outputs = encoder_cbhg(prenet_outputs, is_training)             # [N, T_in, prenet_depth=256]

  log('Encoder Network ...')
  log('  embedding:                 %d' % embedded_inputs.shape[-1])
  log('  prenet out:                %d' % prenet_outputs.shape[-1])
  log('  encoder out:               %d' % encoder_outputs.shape[-1])

  return encoder_outputs


def decoder(inputs, encoder_outputs, is_training, batch_size, mel_targets):
  """ Decoder
  
  Prenet -> Attention RNN
  Postprocessing CBHG

  @param    encoder_outputs   outputs from the encoder wtih shape [N, T_in, prenet_depth=256]
  @param    inputs              int32 Tensor with shape [N, T_in] where N is batch size, T_in is number of
                                steps in the input time series, and values are character IDs
  @param    is_training         flag for training or eval
  @param    batch_size          number of samples per batch
  @param    mel_targets         float32 Tensor with shape [N, T_out, M] where N is batch size, T_out is number
                                of steps in the output time series, M is num_mels, and values are entries in the mel
  @param    output_cell         attention cell
  @param    decoder_init_state  initial state of the decoder

  @return                       linear_outputs, mel_outputs and alignments
  """

  if (is_training):
    helper = TacoTrainingHelper(inputs, mel_targets, hparams.num_mels, hparams.outputs_per_step)
  else:
    helper = TacoTestHelper(batch_size, hparams.num_mels, hparams.outputs_per_step)

  """ Decoder 1 """
  attention_cell = attention_decoder(encoder_outputs)                     # [N, T_in, attention_depth=256]

  # apply prenet before concatanation in AttentionWrapper
  attention_cell = Decoder_Prenet(attention_cell, is_training, hparams.prenet_depths)

  # Concatenate attention context vector and RNN cell output into a 2*attention_depth=512D vector.
  concat_cell = ConcatOutputAndAttentionWrapper(attention_cell)           # [N, T_in, 2*attention_depth=512]

  decoder_cell = tf.nn.rnn_cell.MultiRNNCell([
    tf.contrib.rnn.OutputProjectionWrapper(concat_cell, hparams.decoder_depth),
    tf.nn.rnn_cell.ResidualWrapper(tf.nn.rnn_cell.GRUCell(hparams.decoder_depth)),
    tf.nn.rnn_cell.ResidualWrapper(tf.nn.rnn_cell.GRUCell(hparams.decoder_depth))
  ], state_is_tuple=True)                                                 # [N, T_in, decoder_depth=256]

  # project onto r mel_spectrograms (predict r outputs at each RNN step)
  output_cell = tf.contrib.rnn.OutputProjectionWrapper(decoder_cell, hparams.num_mels * hparams.outputs_per_step)
  decoder_init_state = output_cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)

  
  """ Decoder 2 """
  (decoder_outputs, _), decoder_final_state, _ = tf.contrib.seq2seq.dynamic_decode(
                                                  tf.contrib.seq2seq.BasicDecoder(output_cell, helper, decoder_init_state),
                                                  maximum_iterations=hparams.max_iters
                                                )                         # [N, T_out/r, M*r]

  # reshape outputs to be one output per entry
  mel_outputs = tf.reshape(decoder_outputs, [batch_size, -1, hparams.num_mels]) # [N, T_out, M]

  # post-processing CBHG
  post_outputs = post_cbhg(mel_outputs, hparams.num_mels, is_training)    # [N, T_out, postnet_depth=256]

  linear_outputs = tf.keras.layers.Dense(hparams.num_freq)(post_outputs)  # [N, T_out, F]

  # grab alignments from the final state
  alignments = tf.transpose(decoder_final_state[0].alignment_history.stack(), [1, 2, 0])

  log('Decoder Network ...')
  log('  attention out:             %d' % attention_cell.output_size)
  log('  concat attn & out:         %d' % concat_cell.output_size)
  log('  decoder cell out:          %d' % decoder_cell.output_size)
  log('  decoder out (%d frames):   %d' % (hparams.outputs_per_step, decoder_outputs.shape[-1]))
  log('  decoder out (1 frame):     %d' % mel_outputs.shape[-1])
  log('  postnet out:               %d' % post_outputs.shape[-1])
  log('  linear out:                %d' % linear_outputs.shape[-1])

  return linear_outputs, mel_outputs, alignments
