import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell, OutputProjectionWrapper, ResidualWrapper
from tensorflow.contrib.seq2seq import BasicDecoder, BahdanauAttention, AttentionWrapper

from model.modules import (prenet, encoder_cbhg, post_cbhg,
                          DecoderPrenetWrapper, ConcatOutputAndAttentionWrapper)
from model.helpers import (TacoTrainingHelper, TacoTestHelper)
from constants.hparams import Hyperparams as hparams
from text.character_set import characters
from utils.logger import log


def encoder(inputs, input_lengths, is_training):
  """ Encoder 
  
  Embeddings -> Prenet -> Encoder CBHG

  @param    inputs        int32 Tensor with shape [N, T_in] where N is batch size, T_in is number of
                          steps in the input time series, and values are character IDs
  @param    input_lengths lengths of the inputs
  @param    is_training   flag for training or eval

  @returns                outputs from the encoder
  """

  # Character Embeddings
  embedding_table = tf.get_variable(
                      'embedding',
                      [len(characters), hparams.embed_depth],
                      dtype=tf.float32,
                      initializer=tf.truncated_normal_initializer(stddev=0.5)
                    )

  embedded_inputs = tf.nn.embedding_lookup(embedding_table, inputs)               # [N, T_in, embed_depth=256]

  # Encoder
  prenet_outputs = prenet(embedded_inputs, is_training, hparams.prenet_depths)    # [N, T_in, prenet_depths[-1]=128]
  
  encoder_outputs = encoder_cbhg(prenet_outputs, input_lengths, is_training, hparams.encoder_depth)
                                                                                  # [N, T_in, encoder_depth=256]

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

  # Attention
  attention_cell = AttentionWrapper(
                    GRUCell(hparams.attention_depth),
                    BahdanauAttention(hparams.attention_depth, encoder_outputs),
                    alignment_history=True,
                    output_attention=False
                  )                                                           # [N, T_in, attention_depth=256]
  
  # Apply prenet before concatenation in AttentionWrapper.
  attention_cell = DecoderPrenetWrapper(attention_cell, is_training, hparams.prenet_depths)

  # Concatenate attention context vector and RNN cell output into a 2*attention_depth=512D vector.
  concat_cell = ConcatOutputAndAttentionWrapper(attention_cell)               # [N, T_in, 2*attention_depth=512]

  # Decoder (layers specified bottom to top):
  decoder_cell = MultiRNNCell([
                  OutputProjectionWrapper(concat_cell, hparams.decoder_depth),
                  ResidualWrapper(GRUCell(hparams.decoder_depth)),
                  ResidualWrapper(GRUCell(hparams.decoder_depth))
                ], state_is_tuple=True)                                       # [N, T_in, decoder_depth=256]

  # Project onto r mel spectrograms (predict r outputs at each RNN step):
  output_cell = OutputProjectionWrapper(decoder_cell, hparams.num_mels * hparams.outputs_per_step)

  decoder_init_state = output_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

  (decoder_outputs, _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
                                                  BasicDecoder(output_cell, helper, decoder_init_state),
                                                  maximum_iterations=hparams.max_iters
                                                )                             # [N, T_out/r, M*r]

  # Reshape outputs to be one output per entry
  mel_outputs = tf.reshape(decoder_outputs, [batch_size, -1, hparams.num_mels])   # [N, T_out, M]

  # Add post-processing CBHG:
  post_outputs = post_cbhg(mel_outputs, hparams.num_mels, is_training,            # [N, T_out, postnet_depth=256]
                            hparams.postnet_depth)
  linear_outputs = tf.layers.dense(post_outputs, hparams.num_freq)                # [N, T_out, F]

  # Grab alignments from the final decoder state:
  alignments = tf.transpose(final_decoder_state[0].alignment_history.stack(), [1, 2, 0])

  log('Decoder Network ...')
  log('  attention out:             %d' % attention_cell.output_size)
  log('  concat attn & out:         %d' % concat_cell.output_size)
  log('  decoder cell out:          %d' % decoder_cell.output_size)
  log('  decoder out (%d frames):   %d' % (hparams.outputs_per_step, decoder_outputs.shape[-1]))
  log('  decoder out (1 frame):     %d' % mel_outputs.shape[-1])
  log('  postnet out:               %d' % post_outputs.shape[-1])
  log('  linear out:                %d' % linear_outputs.shape[-1])

  return linear_outputs, mel_outputs, alignments
