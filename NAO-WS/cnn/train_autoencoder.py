from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import math

import copy

import numpy as np
import tensorflow as tf
import encoder
import decoder
import time
import sys
import json

import utils
from params import set_params, Params

SOS = 0
EOS = 0


# Add one more element in the dataset, the decoder source.
# decoder_src is the input to the lstm decoder, which uses the output of the last step.
# Since the first step doesn't have a last step output, we use SOS (start of sequence) = 0 as the input.
def preprocess(encoder_src, decoder_tgt):  # src:sequence tgt:performance
    sos_id = tf.constant([SOS])
    decoder_src = tf.concat([sos_id, decoder_tgt[:-1]], axis=0)
    return encoder_src, decoder_src, decoder_tgt


def input_fn(encoder_input, decoder_target, mode, batch_size, num_epochs=1, symmetry=False):
    shape = np.array(encoder_input).shape
    N = shape[0]
    source_length = shape[1]
    # converting numpy arrays to tf Dataset.
    encoder_input = tf.convert_to_tensor(encoder_input, dtype=tf.int32)
    encoder_input = tf.data.Dataset.from_tensor_slices(encoder_input)
    decoder_target = tf.convert_to_tensor(decoder_target, dtype=tf.int32)
    decoder_target = tf.data.Dataset.from_tensor_slices(decoder_target)
    dataset = tf.data.Dataset.zip((encoder_input, decoder_target))
    dataset = dataset.shuffle(buffer_size=N)

    def generate_symmetry(encoder_src, decoder_src, decoder_tgt):
        a = tf.random_uniform([], 0, 5, dtype=tf.int32)
        b = tf.random_uniform([], 0, 5, dtype=tf.int32)
        cell_seq_length = source_length // 2
        assert source_length in [40, 60]
        if source_length == 40:
            encoder_src = tf.concat(
                [encoder_src[:4 * a], encoder_src[4 * a + 2:4 * a + 4], encoder_src[4 * a:4 * a + 2],
                 encoder_src[4 * (a + 1):cell_seq_length + 4 * b],
                 encoder_src[cell_seq_length + 4 * b + 2:cell_seq_length + 4 * b + 4],
                 encoder_src[cell_seq_length + 4 * b:cell_seq_length + 4 * b + 2],
                 encoder_src[cell_seq_length + 4 * (b + 1):]], axis=0)

        else:  # source_length=60
            encoder_src = tf.concat(
                [encoder_src[:6 * a], encoder_src[6 * a + 3:6 * a + 6], encoder_src[6 * a:6 * a + 3],
                 encoder_src[6 * (a + 1):cell_seq_length + 6 * b],
                 encoder_src[cell_seq_length + 6 * b + 3:cell_seq_length + 6 * b + 6],
                 encoder_src[cell_seq_length + 6 * b:cell_seq_length + 6 * b + 3],
                 encoder_src[cell_seq_length + 6 * (b + 1):]], axis=0)
        decoder_tgt = encoder_src
        return encoder_src, decoder_src, decoder_tgt

    dataset = dataset.map(preprocess)
    if symmetry:
        dataset = dataset.map(generate_symmetry)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(10)
    iterator = dataset.make_one_shot_iterator()
    encoder_input, decoder_input, decoder_target = iterator.get_next()
    assert encoder_input.shape.ndims == 2
    assert decoder_input.shape.ndims == 2
    assert decoder_target.shape.ndims == 2
    return encoder_input, decoder_input, decoder_target


def get_train_ops(encoder_train_input, decoder_train_input, decoder_train_target,
                  reuse=tf.AUTO_REUSE):
    with tf.variable_scope('EPD', reuse=reuse):
        my_encoder = encoder.Encoder(encoder_train_input, tf.estimator.ModeKeys.TRAIN, 'Encoder', reuse)
        encoder_outputs = my_encoder.encoder_outputs
        encoder_state = my_encoder.arch_emb
        encoder_state.set_shape([None, Params.decoder_hidden_size])
        encoder_state = tf.contrib.rnn.LSTMStateTuple(encoder_state, encoder_state)
        encoder_state = (encoder_state,) * Params.decoder_num_layers
        my_decoder = decoder.DecoderModel(encoder_outputs, encoder_state, decoder_train_input, decoder_train_target,
                                          tf.estimator.ModeKeys.TRAIN, 'Decoder', reuse)
        decoder_loss = my_decoder.loss
        cross_entropy = decoder_loss

        total_loss = decoder_loss + Params.weight_decay * tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

        tf.summary.scalar('training_loss', total_loss)

        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.constant(Params.controller_lr)
        if Params.optimizer == "sgd":
            learning_rate = tf.cond(
                global_step < Params.start_decay_step,
                lambda: learning_rate,
                lambda: tf.train.exponential_decay(
                    learning_rate,
                    (global_step - Params.start_decay_step),
                    Params.decay_steps,
                    Params.decay_factor,
                    staircase=True),
                name="calc_learning_rate")
            opt = tf.train.GradientDescentOptimizer(learning_rate)
        elif Params.optimizer == "adam":
            assert float(Params.controller_lr) <= 0.001, "! High Adam learning rate %g" % Params.controller_lr
            opt = tf.train.AdamOptimizer(learning_rate)
        elif Params.optimizer == 'adadelta':
            opt = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
        tf.summary.scalar("learning_rate", learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            gradients, variables = zip(*opt.compute_gradients(total_loss))
            grad_norm = tf.global_norm(gradients)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, Params.max_gradient_norm)
            train_op = opt.apply_gradients(
                zip(clipped_gradients, variables), global_step=global_step)

        return cross_entropy, total_loss, learning_rate, train_op, global_step, grad_norm


# encoder target is the list of performances of the architectures
def train(encoder_input):
    decoder_target = copy.copy(encoder_input)
    with tf.Graph().as_default():
        tf.logging.info('Training Encoder-Predictor-Decoder')
        tf.logging.info('Preparing data')
        shape = np.array(encoder_input).shape
        N = shape[0]
        encoder_train_input, decoder_train_input, decoder_train_target = input_fn(
            encoder_input,
            decoder_target,
            'train',
            Params.controller_batch_size,
            None,
            Params.symmetry
        )
        tf.logging.info('Building model')
        train_cross_entropy, train_loss, learning_rate, train_op, global_step, grad_norm = get_train_ops(
            encoder_train_input, decoder_train_input, decoder_train_target)
        saver = tf.train.Saver(max_to_keep=10)
        checkpoint_saver_hook = tf.train.CheckpointSaverHook(
            Params.autoencoder_model_dir, save_steps=Params.batches_per_epoch * Params.save_frequency,
            saver=saver)
        hooks = [checkpoint_saver_hook]
        merged_summary = tf.summary.merge_all()
        tf.logging.info('Starting Session')
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.train.SingularMonitoredSession(
                config=config, hooks=hooks, checkpoint_dir=Params.autoencoder_model_dir) as sess:
            writer = tf.summary.FileWriter(Params.autoencoder_model_dir, sess.graph)
            start_time = time.time()
            for step in range(Params.train_epochs * Params.batches_per_epoch):
                run_ops = [
                    train_cross_entropy,
                    train_loss,
                    learning_rate,
                    train_op,
                    global_step,
                    grad_norm,
                    merged_summary,
                ]
                train_cross_entropy_v, train_loss_v, learning_rate_v, _, global_step_v, gn_v, summary = sess.run(
                    run_ops)

                writer.add_summary(summary, global_step_v)

                epoch = (global_step_v + 1) // Params.batches_per_epoch

                curr_time = time.time()
                if (global_step_v + 1) % 100 == 0:
                    log_string = "epoch={:<6d} ".format(epoch)
                    log_string += "step={:<6d} ".format(global_step_v + 1)
                    log_string += "cross_entropy={:<6f} ".format(train_cross_entropy_v)
                    log_string += "loss={:<6f} ".format(train_loss_v)
                    log_string += "learning_rate={:<8.4f} ".format(learning_rate_v)
                    log_string += "|gn|={:<8.4f} ".format(gn_v)
                    log_string += "mins={:<10.2f}".format((curr_time - start_time) / 60)
                    tf.logging.info(log_string)


def main(unused_argv):
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    arch_pool = utils.generate_arch(100000, 5, 5)
    branch_length = 40 // 2 // 5 // 2
    encoder_input = list(
        map(lambda x: utils.parse_arch_to_seq(x[0], branch_length) + utils.parse_arch_to_seq(x[1], branch_length),
            arch_pool))
    Params.batches_per_epoch = math.ceil(len(encoder_input) / Params.controller_batch_size)
    train(encoder_input)


if __name__ == '__main__':
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
    tf.logging.set_verbosity(tf.logging.INFO)
    unparsed = set_params()
    tf.app.run(argv=[sys.argv[0]] + unparsed)
