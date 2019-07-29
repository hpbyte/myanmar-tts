from keras.layers import Input, Embedding, concatenate, RepeatVector, Dense, Reshape
from keras.models import Model
from model.modules import *

def Tacotron(n_mels, r, k1, k2, nb_char_max, embedding_size, mel_time_length, mag_time_length, n_fft, vocab):
  """
  Tacotron Model
  """
  # *** Encoder ***
  encoder_input = Input(shape=(nb_char_max,))
  embedded = Embedding(input_dim=len(vocab), output_dim=embedding_size, input_length=nb_char_max)(encoder_input)
  encoder_prenet = Pre_Net(embedded)
  encoder_cbhg = CBHG(encoder_prenet, k1)

  # *** Decoder Part 1 Pre-net ***
  decoder_input = Input(shape=(None, n_mels))
  decoder_prenet = Pre_Net(decoder_input)
  attention_rnn_output = Attention_RNN()(decoder_prenet)

  # *** Attention ***
  attention_rnn_output_repeated = RepeatVector(nb_char_max)(attention_rnn_output)
  attention_context = Attention_Context(encoder_cbhg, attention_rnn_output_repeated)

  context_shape1 = int(attention_context.shape[1])
  context_shape2 = int(attention_context.shape[2])
  attention_rnn_output_reshaped = Reshape((context_shape1, context_shape2))(attention_rnn_output)

  # *** Decoder Part 2 ***
  decoder_rnn_input = concatenate([attention_context, attention_rnn_output_reshaped])

  decoder_rnn_projected_input = Dense(256)(decoder_rnn_input)

  decoder_rnn_output = Decoder_RNN(decoder_rnn_projected_input)

  mel_hat = Dense(mel_time_length * n_mels * r)(decoder_rnn_output)
  mel_hat_ = Reshape((mel_time_length, n_mels * r))(mel_hat)

  def slice(x):
    return x[:, :, -n_mels:]

  mel_hat_last_frame = Lambda(slice)(mel_hat_)
  post_process_output = CBHG(mel_hat_last_frame, k2, for_encoder=False)

  z_hat = Dense(mag_time_length * (1 + n_fft // 2))(post_process_output)
  z_hat_ = Reshape((mag_time_length, (1 + n_fft // 2)))(z_hat)

  model = Model(inputs=[encoder_input, decoder_input], outputs=[mel_hat_, z_hat_])

  return model
