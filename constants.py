# === Text ===
NB_CHARS_MAX = 200        # max size of input text data

# === Audio ===
REF_DB = 20
MAX_DB = 100
SAMPLING_RATE = 16000
r = 5                     # reduction factor
N_FFT = 1024              # no. of Fourier Transform points
PREEMPHASIS = 0.97        # parameter of preemphasis technique that gives more importance to high frequencies in the signal
WINDOW_TYPE = 'hann'      # type of window used to computer Fourier Transform
FRAME_LENGTH = 0.05       # length of window in seconds
FRAME_SHIFT = 0.0125      # temporal shift in seconds
N_MEL = 80                # no. of mel bands
HOP_LENGTH = int(SAMPLING_RATE * FRAME_SHIFT)
WIN_LENGTH = int(SAMPLING_RATE * FRAME_LENGTH)
MAX_MEL_TIME_LENGTH = 400 # max size of the time dimension of the mel spectrogram
MAX_MAG_TIME_LENGTH = 850 # max size of the time dimension of the spectrogram

# === Training Model ===
N_ITER = 60               # no. of iteration
K1 = 16                   # size of the convolution bank in the encoder CBHG
K2 = 8                    # size of the convolution bank in the decoder CBHG
BATCH_SIZE = 32
NB_EPOCHS = 50
EMBEDDING_SIZE = 256

# === Others ===
TRAIN_SET_RATIO = 0.9
BASE_DIR = '~/Projects/text-to-speech/tts.mm-tacotron/data'