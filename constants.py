TRAIN_SET_RATIO = 0.9
# max size of input text data
NB_CHARS_MAX = 200
# no. of Fourier Transform points
N_FFT = 1024
# parameter of preemphasis technique that gives more importance to high frequencies in the signal
PREEMPHASIS = 0.97
# sampling rate
SAMPLING_RATE = 16000
# type of window used to computer Fourier Transform
WINDOW_TYPE = 'hann'
# length of window in seconds
FRAME_LENGTH = 0.05
# temporal shift in seconds
FRAME_SHIFT = 0.0125
# 
HOP_LENGTH = int(SAMPLING_RATE * FRAME_SHIFT)
# 
WIN_LENGTH = int(SAMPLING_RATE * FRAME_LENGTH)
# no. of mel bands
N_MEL = 80
# reduction factor
r = 5
# 
REF_DB = 20
# max decibel
MAX_DB = 100
# max size of the time dimension of the mel spectrogram
MAX_MEL_TIME_LENGTH = 200
# max size of the time dimension of the spectrogram
MAX_MAG_TIME_LENGTH = 850
# no. of iteration
N_ITER = 50

# base path
BASE_DIR = '~/Documents/tacotron/'