from sklearn.externals import joblib
from keras.optimizers import Adam

from constants import *
from model.tacotron import Tacotron

# import prepared data
mel_spectro_training = joblib.load('data/LJSpeech/mel_spectro_training.pkl')
spectro_training = joblib.load('data/LJSpeech/spectro_training.pkl')
decoder_input_training = joblib.load('data/LJSpeech/decoder_input_training.pkl')

text_input_training = joblib.load('data/LJSpeech/text_input_training.pkl')
vocab = joblib.load('data/LJSpeech/vocabulary_id.pkl')

model = Tacotron(N_MEL, r, K1, K2, NB_CHARS_MAX, EMBEDDING_SIZE, MAX_MEL_TIME_LENGTH, MAX_MAG_TIME_LENGTH, N_FFT, vocab)

optimizer = Adam()

model.compile(optimizer=optimizer, loss=['mean_absolute_error', 'mean_absolute_error'])

training_history = model.fit(
                    [text_input_training, decoder_input_training],
                    [mel_spectro_training, spectro_training],
                    epochs=NB_EPOCHS, batch_size=BATCH_SIZE, verbose=1, validation_split=0.15
                  )

joblib.dump(training_history.history, 'results/training.history.pkl')

model.save('results/model.h5')
