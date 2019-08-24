import os
import math
import subprocess
import time
from datetime import datetime
import traceback
import argparse

import tensorflow as tf

from model.feeder import DataFeeder
from model.tacotron import Tacotron
from signal_proc import audio
from text.tokenizer import sequence_to_text
import constants.hparams as hparams
from utils import logger, plotter, ValueWindow


def add_stats(model):
  with tf.compat.v1.variable_scope('stats'):
    tf.summary.histogram('linear_outputs', model.linear_outputs)
    tf.summary.histogram('linear_targets', model.linear_targets)
    tf.summary.histogram('mel_outputs', model.mel_outputs)
    tf.summary.histogram('mel_targets', model.mel_targets)
    tf.summary.scalar('loss_mel', model.mel_loss)
    tf.summary.scalar('loss_linear', model.linear_loss)
    tf.summary.scalar('learning_rate', model.learning_rate)
    tf.summary.scalar('loss', model.loss)
    gradient_norms = [tf.norm(grad) for grad in model.gradients]
    tf.summary.histogram('gradient_norm', gradient_norms)
    tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms))
    return tf.summary.merge_all()


def time_string():
  return datetime.now().strftime('%Y-%m-%d %H:%M')


def train(log_dir, args):
  checkpoint_path = os.path.join(log_dir, 'model.ckpt')
  input_path = os.path.join(args.base_dir, 'training/train.txt')

  logger.log('Checkpoint path: %s' % checkpoint_path)
  logger.log('Loading training data from: %s' % input_path)

  # set up DataFeeder
  coordi = tf.train.Coordinator()
  with tf.compat.v1.variable_scope('data_feeder'):
    feeder = DataFeeder(coordi, input_path)

  # set up Model
  global_step = tf.Variable(0, name='global_step', trainable=False)
  with tf.compat.v1.variable_scope('model'):
    model = Tacotron()
    model.init(feeder.inputs, feeder.input_lengths, mel_targets=feeder.mel_targets, linear_targets=feeder.linear_targets)
    model.add_loss()
    model.add_optimizer(global_step)
    stats = add_stats(model)

  # book keeping
  step = 0
  loss_window = ValueWindow(100)
  time_window = ValueWindow(100)
  saver = tf.compat.v1.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

  # start training already!
  with tf.compat.v1.Session() as sess:
    try:
      summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
      
      # initialize parameters
      sess.run(tf.compat.v1.global_variables_initializer())

      # if requested, restore from step
      if (args.restore_step):
        restore_path = '%s-%d' % (checkpoint_path, args.restore_step)
        saver.restore(sess, restore_path)
        logger.log('Resuming from checkpoint: %s at commit: %s' % restore_path)
      else:
        logger.log('Starting a new training!')

      feeder.start_in_session(sess)

      while not coordi.should_stop():
        start_time = time.time()

        step, loss, opt = sess.run([global_step, model.loss, model.optimize])

        time_window.append(time.time() - start_time)
        loss_window.append(loss)

        msg = 'Step %-7d [%.03f sec/step, loss=%.05f, avg_loss=%.05f]' % (
                  step, time_window.average, loss, loss_window.average)

        logger.log(msg)

        if loss > 100 or math.isnan(loss):
          # bad situation
          logger.log('Loss exploded to %.05f at step %d!' % (loss, step))
          raise Exception('Loss Exploded')

        if step % args.summary_interval == 0:
          # it's time to write summary
          logger.log('Writing summary at step: %d' % step)
          summary_writer.add_summary(sess.run(stats), step)

        if step % args.checkpoint_interval == 0:
          # it's time to save a checkpoint
          logger.log('Saving checkpoint to: %s-%d' % (checkpoint_path, step))
          saver.save(sess, checkpoint_path, global_step=step)
          logger.log('Saving audio and alignment...')

          input_seq, spectrogram, alignment = sess.run([
                                              model.inputs[0], model.linear_outputs[0], model.alignments[0]
                                            ])

          # convert spectrogram to waveform
          waveform = audio.spectrogram_to_wav(spectrogram.T)
          # save it
          audio.save_audio(waveform, os.path.join(log_dir, 'step-%d-audio.wav' % step))

          plotter.plot_alignment(
            alignment, 
            os.path.join(log_dir, 'step-%d-align.png' % step),
            info='%s, %s, step=%d, loss=%.5f' % (args.model, time_string(), step, loss)
          )

          logger.log('Input: %s' % sequence_to_text(input_seq))
        
    except Exception as e:
      logger.log('Exiting due to exception %s' % e)
      traceback.print_exc()
      coordi.request_stop(e)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', default=os.path.expanduser('~/Documents/tacotron'))
  parser.add_argument('--input', default='training/train.txt')
  parser.add_argument('--restore_step', type=int, help='Global step to restore from checkpoint.')
  parser.add_argument('--summary_interval', type=int, default=100, help='Steps between running summary ops.')
  parser.add_argument('--checkpoint_interval', type=int, default=1000, help='Steps between writing checkpoints.')
  args = parser.parse_args()

  log_dir = os.path.join(args.base_dir, args.input)
  os.makedirs(log_dir, exist_ok=True)
  logger.init(os.path.join(log_dir, 'train.log'))

  train(log_dir, args)


if __name__ == "__main__":
  main()