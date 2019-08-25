class Hyperparams:
  """ Hyper parameters """
  # Signal
  num_mels = 80
  num_freq = 1025
  sample_rate = 20000
  frame_length = 0.05
  frame_shift = 0.0125
  preemphasis = 0.97
  min_db = -100
  ref_db = 20

  # parameters
  n_fft = (num_freq - 1) * 2
  hop_length = int(frame_shift * sample_rate)
  win_length = int(frame_length * sample_rate)

  max_iters = 200
  griffin_lim_iters = 60
  power = 1.5

  # for training
  batch_size = 32
  learning_rate_decay = True
  initial_lr = 0.002
  adam_beta_1 = 0.9
  adam_beta_2 = 0.999

  # Model
  outputs_per_step = 5
  embed_depth = 256
  prenet_depths = [256, 128]
  encoder_depth = 256
  postnet_depth = 256
  attention_depth = 256
  decoder_depth = 256
