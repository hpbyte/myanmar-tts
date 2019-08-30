import io

import numpy as np
import tensorflow as tf
from librosa import effects

from model.tacotron import Tacotron
from signal_proc import audio
from text.tokenizer import text_to_sequence
from constants.hparams import Hyperparams as hparams


class Synthesizer():
  """ Synthesizer """

  def init(self, checkpoint_path):
    """ Initialize Synthesizer 
    
    @type   checkpoint_path   str
    @param  checkpoint_path   path to checkpoint to be restored    
    """
    print('Constructing Tacotron Model ...')

    inputs = tf.compat.v1.placeholder(tf.int32, [1, None], 'inputs')
    input_lengths = tf.compat.v1.placeholder(tf.int32, [1], 'input_lengths')

    with tf.compat.v1.variable_scope('model'):
      self.model = Tacotron()
      self.model.init(inputs, input_lengths)
      self.wav_output = audio.spectrogram_to_wav_tf(self.model.linear_outputs[0])

    print('Loading checkpoint: %s' % checkpoint_path)
    self.session = tf.compat.v1.Session()
    self.session.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver()
    saver.restore(self.session, checkpoint_path)


  def synthesize(self, text):
    """ Convert the text into synthesized speech 
    
    @type   text    str
    @param  text    text to be synthesized

    @rtype          object
    @return         synthesized speech
    """

    seq = text_to_sequence(text)

    feed_dict = {
      self.model.inputs: [np.asarray(seq, dtype=np.int32)],
      self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32)
    }

    wav = self.session.run(self.wav_output, feed_dict=feed_dict)
    wav = audio.inv_preemphasis(wav)
    wav = wav[:audio.find_endpoint(wav)]
    out = io.BytesIO()
    audio.save_audio(wav, out)

    return out.getvalue()


