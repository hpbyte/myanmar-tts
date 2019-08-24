import tensorflow as tf

# Core Modules

def prenet(inputs, is_training, scope=None):
  """ Pre-Net

  FC-256-ReLU -> Dropout(0.5) -> FC-128-ReLU -> Dropout(0.5)
  """
  prenet = inputs
  drop_rate = 0.5 if is_training else 0.0

  with tf.compat.v1.variable_scope(scope or 'prenet'):
    prenet = tf.keras.layers.Dense(256, activation='relu')(prenet)
    prenet = tf.keras.layers.Dropout(drop_rate)(prenet, training=is_training)
    prenet = tf.keras.layers.Dense(128, activation='relu')(prenet)
    prenet = tf.keras.layers.Dropout(drop_rate)(prenet, training=is_training)

    return prenet


def conv1d(inputs, kernel_size, filters, activation, is_training, scope):
  """ Conv1d with batch normalization """
  with tf.compat.v1.variable_scope(scope):
    conv1d = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation=activation, padding='same')(inputs)
    conv1d = tf.keras.layers.BatchNormalization()(conv1d, training=is_training)

    return conv1d


def conv1d_bank(inputs, K, is_training):
  """ 1-D Convolution Bank

  K-layers of Conv1D filters to model Unigrams, Bigrams and so on
  """
  with tf.compat.v1.variable_scope('conv1d_bank'):
    bank = tf.keras.layers.Concatenate(axis=-1)([conv1d(inputs, k, 128, tf.nn.relu, is_training, 'conv1d_%d' % k) for k in range(1, K+1)])
    return bank


def highwaynet(highway_input, nb_layers, depth):
  """ HighwayNet """
  highway_output = highway_input
  for i in range(nb_layers):
    with tf.compat.v1.variable_scope('highway_%d' % (i+1)):
      H = tf.keras.layers.Dense(depth, activation='relu', name='H')(highway_input)
      T = tf.keras.layers.Dense(depth, activation='sigmoid', name='T', bias_initializer=tf.constant_initializer(-1.0))(highway_input)

      highway_output = H * T + highway_input * (1.0 - T)

  return highway_output


def cbhg(inputs, is_training, scope, K, projections):
  """ CBHG module (Conv1D Bank, Highway Net, Bidirectional GRU)

  a powerful module for extracting representations from sequences
  """
  with tf.compat.v1.variable_scope(scope):
    # Conv1D bank 
    #   encoder :  K=16, conv-k-128-ReLU
    #   post    :  K=8,  conv-k-128-ReLU
    conv_outputs = conv1d_bank(inputs, K, is_training)

    # Maxpooling 
    #   stride=1, width=2
    maxpool_output = tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='same')(conv_outputs)

    # Two projections layers
    #   encoder :  conv-3-128-ReLU → conv-3-128-Linear
    #   post    :  conv-3-256-ReLU → conv-3-80-Linear
    proj1_output = conv1d(maxpool_output, 3, projections[0], tf.nn.relu, is_training, 'proj_1')
    proj2_output = conv1d(proj1_output, 3, projections[1], None, is_training, 'proj_2')

    # Residual connection
    highway_input = proj2_output + inputs

    # 4-layer HighwayNet
    highway_output = highwaynet(highway_input, 4, 128)

    # Bidirectional GRU 128 cells
    outputs = tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(128, return_sequences=True),
                backward_layer=tf.keras.layers.GRU(128, return_sequences=True, go_backwards=True)
              )(highway_output)

    return outputs


def encoder_cbhg(inputs, is_training):
  """ Encoder CBHG """
  input_channels = inputs.get_shape()[2]
  return cbhg(inputs, is_training, scope='encoder_cbhg', K=16, projections=[128, input_channels])


def post_cbhg(inputs, input_dim, is_training):
  """ Post-processing net CBHG """
  return cbhg(inputs, is_training, scope='post_cbhg', K=8, projections=[256, input_dim])


def attention_decoder(inputs):
  """ Attention Decoder """
  attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(256, inputs)
  decoder_cell = tf.keras.layers.GRUCell(256)
  attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                                        decoder_cell, attention_mechanism, 
                                        alignment_history=True, output_attention=False
                                      )

  return attention_cell


# Other Modules

class Decoder_Prenet(tf.compat.v1.nn.rnn_cell.RNNCell):
  """Runs RNN inputs through a prenet before sending them to the cell."""
  def __init__(self, cell, is_training, layer_sizes):
    super(Decoder_Prenet, self).__init__()
    self._cell = cell
    self._is_training = is_training
    self._layer_sizes = layer_sizes

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def call(self, inputs, state):
    prenet_out = prenet(inputs, self._is_training, scope='decoder_prenet')
    return self._cell(prenet_out, state)

  def zero_state(self, batch_size, dtype):
    return self._cell.zero_state(batch_size, dtype)


class ConcatOutputAndAttentionWrapper(tf.compat.v1.nn.rnn_cell.RNNCell):
  """Concatenates RNN cell output with the attention context vector.

  This is expected to wrap a cell wrapped with an AttentionWrapper constructed with
  attention_layer_size=None and output_attention=False. Such a cell's state will include an
  "attention" field that is the context vector.
  """
  def __init__(self, cell):
    super(ConcatOutputAndAttentionWrapper, self).__init__()
    self._cell = cell

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size + self._cell.state_size.attention

  def call(self, inputs, state):
    output, res_state = self._cell(inputs, state)
    return tf.concat([output, res_state.attention], axis=-1), res_state

  def zero_state(self, batch_size, dtype):
    return self._cell.zero_state(batch_size, dtype)

