import tensorflow as tf

hparams = tf.contrib.training.HParams(
  # Signal
  num_mels=80,
  num_freq=1025,
  sample_rate=20000,
  frame_length_ms=50,
  frame_shift_ms=12.5,
  preemphasis=0.97,
  min_level_db=-100,
  ref_level_db=20,

  max_iters=200,
  griffin_lim_iters=60,
  power=1.5,
)
