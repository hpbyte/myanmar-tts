import tensorflow as tf
from keras.layers import (Dense, Dropout, Conv1D, BatchNormalization, Bidirectional,
                          MaxPooling1D, GRUCell)


def prenet(inputs, is_training, layer_sizes, scope=None):
  """ Pre-Net

  FC-256-ReLU -> Dropout(0.5) -> FC-128-ReLU -> Dropout(0.5)
  """
  prenet = inputs
  drop_rate = 0.5 if is_training else 0.0

  with tf.variable_scope(scope or 'prenet'):
    prenet = Dense(256, activation='relu')(prenet)
    prenet = Dropout(drop_rate, training=is_training)(prenet)
    prenet = Dense(128, activation='relu')(prenet)
    prenet = Dropout(drop_rate, training=is_training)(prenet)

  return prenet


def conv1d(inputs, kernel_size, filters, activation, is_training, scope):
  """ Conv1d with batch normalization """
  with tf.variable_scope(scope):
    conv1d = Conv1D(filters=filters, kernel_size=kernel_size, activation=activation, padding='same')(inputs)
    conv1d = BatchNormalization(training=is_training)(conv1d)

    return conv1d


def conv1d_bank(inputs, K, is_training):
  """ 1-D Convolution Bank

  K-layers of Conv1D filters to model Unigrams, Bigrams and so on
  """
  with tf.variable_scope('conv1d_bank'):
    bank = tf.concat(
      [conv1d(inputs, k, 128, tf.nn.relu, is_training, 'conv1d_%d' % k) for k in range(1, K+1)],
      axis=-1
    )
    return bank


def highwaynet(highway_input, nb_layers, depth):
  """ HighwayNet """
  for i in range(nb_layers):
    with tf.variable_scope('highway_%d' % (i+1)):
      H = Dense(depth, activation='relu', name='H')(highway_input)
      T = Dense(depth, activation='sigmoid', name='T', bias_initializer=tf.constant_initializer(-1.0))(highway_input)

      highway_output = H * T + highway_input * (1.0 - T)

  return highway_output


def cbhg(inputs, is_training, scope, K, projections):
  """ CBHG module (Conv1D Bank, Highway Net, Bidirectional GRU)

  a powerful module for extracting representations from sequences
  """
  with tf.variable_scope(scope):
    # Conv1D bank 
    #   encoder :  K=16, conv-k-128-ReLU
    #   post    :  K=8,  conv-k-128-ReLU
    conv_outputs = conv1d_bank(inputs, K, is_training)

    # Maxpooling 
    #   stride=1, width=2
    maxpool_output = MaxPooling1D(pool_size=2, strides=1, padding='same')(conv_outputs)

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
    outputs = Bidirectional(GRUCell(128), backward_layer=GRUCell(128))(highway_output)

    return outputs


def encoder_cbhg(inputs, is_training):
  """ Encoder CBHG """
  input_channels = inputs.get_shape()[2]
  return cbhg(inputs, is_training, scope='encoder_cbhg', K=16, projections=[128, input_channels])


def post_cbhg(inputs, input_dim, is_training):
  """ Post-processing net CBHG """
  return cbhg(inputs, is_training, scope='post_cbhg', K=8, projections=[256, input_dim])
