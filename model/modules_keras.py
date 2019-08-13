import keras.backend as K
import keras.initializers as k_init
from keras.layers import (Conv1D, Dense, Activation, MaxPooling1D, 
                          Add, Concatenate, Bidirectional, GRU, Dropout,
                          BatchNormalization, Lambda, Dot, Multiply)

def Pre_Net(input_data):
  """
  Pre-Net

  FC-256-ReLU -> Dropout(0.5) -> FC-128-ReLU -> Dropout(0.5)
  """
  prenet = Dense(256)(input_data)
  prenet = Activation('relu')(prenet)
  prenet = Dropout(0.5)(prenet)
  prenet = Dense(128)(prenet)
  prenet = Activation('relu')(prenet)
  prenet = Dropout(0.5)(prenet)

  return prenet


def Conv1D_Bank(K_, input_data):
  """
  1-D Convolution Bank

  K-layers of Conv1D filters to model Unigrams, Bigrams and so on
  """
  conv = Conv1D(filters=128, kernel_size=1, strides=1, padding='same')(input_data)
  conv = BatchNormalization()(conv)
  conv = Activation('relu')(conv)

  for k_ in range(2, K_+1):
    conv = Conv1D(filters=128, kernel_size=k_, strides=1, padding='same')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

  return conv


def Highway_Net(highway_input, nb_layers, activation = 'relu', bias = -3):
  """
  Highway Network
  """
  dim = K.int_shape(highway_input)[-1]
  initial_bias = k_init.Constant(bias)

  for n in range(nb_layers):
    H = Dense(units=dim, bias_initializer=initial_bias)(highway_input)
    H = Activation('sigmoid')(H)
    carry_gate = Lambda(lambda x: 1.0 - x, output_shape=(dim,))(H)
    transform_gate = Dense(units=dim)(highway_input)
    transform_gate = Activation(activation)(transform_gate)
    transformed = Multiply()([H, transform_gate])
    carried = Multiply()([carry_gate, highway_input])
    highway_output = Add()([transformed, carried])

  return highway_output


def CBHG(input_data, K_CBHG, for_encoder = True):
  """
  CBHG module (Conv1D Bank, Highway Net, Bidirectional GRU)
  """
  # Conv1D Bank
  conv1d_bank = Conv1D_Bank(K_CBHG, input_data)
  # Max Pooling 
  conv1d_bank = MaxPooling1D(pool_size=2, strides=1, padding='same')(conv1d_bank)
  # Conv1D
  conv1d_bank = Conv1D(filters=128, kernel_size=3, strides=1, padding='same')(conv1d_bank)
  # Batch Normalization
  conv1d_bank = BatchNormalization()(conv1d_bank)
  # Residual Connection
  residual = Add()([conv1d_bank, input_data])
  # Highway Net
  highway_net = Highway_Net(residual, 4, activation='relu')

  if for_encoder:
    # for encoder
    CBHG = Bidirectional(GRU(128, return_sequences=True))(highway_net)
  else:
    # for decoder
    CBHG = Bidirectional(GRU(128))(highway_net)

  return CBHG


def Attention_RNN():
  """
  Attention RNN

  1-layer GRU with 256 units
  """
  return GRU(256)


def Decoder_RNN(input_data):
  """
  Decoder RNN

  2-layer GRU with vertical residual connections
  ***
  have to define return_sequences=True for the first GRU layer -> an output will be returned for each timestep
  if don't, the 1st GRU returns only one output for the entire input sequence while the 2nd GRU expects a sequence as input
  ***
  """
  rnn1 = GRU(256, return_sequences=True)(input_data)

  inp2 = Add()([input_data, rnn1])
  rnn2 = GRU(256)(inp2)

  decoder_rnn = Add()([inp2, rnn2])

  return decoder_rnn


def Attention_Context(encoder_output, attention_rnn_output):
  """
  Attention Mechanism

  (CBHG_encoder_ouput + RNN_output) -> tanh -> relu -> softmax -> * CBHG_encoder_output
  """
  attention_input = Concatenate(axis=-1)([encoder_output, attention_rnn_output])
  e = Dense(10, activation='tanh')(attention_input)
  energies = Dense(1, activation='relu')(e)
  attention_weights = Activation('softmax')(energies)
  context = Dot(axes=1)([attention_weights, encoder_output])

  return context
