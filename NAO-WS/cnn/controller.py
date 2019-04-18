from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import argparse
import math
import sys

import numpy as np
import tensorflow as tf
import encoder
import decoder
import time

import utils
from params import Params, set_params

SOS = 0
EOS = 0


def encode(original_encoder_input):
    N = np.array(original_encoder_input).shape[0]
    with tf.Graph().as_default():
        tf.logging.info(
            'Generating new architectures using gradient descent with step size {}'.format(Params.predict_lambda))
        tf.logging.info('Preparing data')
        encoder_input = tf.convert_to_tensor(original_encoder_input, dtype=tf.int32)
        encoder_input = tf.data.Dataset.from_tensor_slices(encoder_input)
        encoder_input = encoder_input.batch(N)
        iterator = encoder_input.make_one_shot_iterator()
        encoder_input = iterator.get_next()

        with tf.variable_scope('EPD', reuse=tf.AUTO_REUSE):
            my_encoder = encoder.Encoder(encoder_input, tf.estimator.ModeKeys.PREDICT, 'Encoder', tf.AUTO_REUSE)
            embed = my_encoder.predict()

        tf.logging.info('Starting Session')
        config = tf.ConfigProto(allow_soft_placement=True)
        arch_embed, encoder_outputs = [], []
        with tf.train.SingularMonitoredSession(
                config=config, checkpoint_dir=Params.autoencoder_model_dir) as sess:
            a, b = sess.run(embed)
            arch_embed.append(a)
            encoder_outputs.append(b)
        arch_embed = np.array(arch_embed)
        encoder_outputs = np.array(encoder_outputs)
        encoder_outputs = encoder_outputs.transpose([0, 2, 1, 3])
        return arch_embed.reshape((arch_embed.shape[0] * arch_embed.shape[1],) + arch_embed.shape[2:]),\
               encoder_outputs.reshape((encoder_outputs.shape[0] * encoder_outputs.shape[1],) + encoder_outputs.shape[2:])


def decode(decoder_inputs, new_arch_outputs):
    with tf.Graph().as_default():
        tf.logging.info(
            'Generating new architectures using gradient descent with step size {}'.format(Params.predict_lambda))
        tf.logging.info('Preparing data')
        n = new_arch_outputs.shape[0]
        decoder_inputs = tf.constant([SOS] * n, dtype=tf.int32)
        new_arch_outputs = tf.convert_to_tensor(new_arch_outputs, dtype=tf.float32)
        if Params.time_major:
            new_arch_outputs = tf.transpose(new_arch_outputs, [1, 0, 2])
        new_arch_outputs = tf.nn.l2_normalize(new_arch_outputs, dim=-1)
        if Params.time_major:
            new_arch_emb = tf.reduce_mean(new_arch_outputs, axis=0)
        else:
            new_arch_emb = tf.reduce_mean(new_arch_outputs, axis=1)
        new_arch_emb = tf.nn.l2_normalize(new_arch_emb, dim=-1)

        encoder_state = new_arch_emb
        encoder_state.set_shape([None, Params.decoder_hidden_size])
        encoder_state = tf.contrib.rnn.LSTMStateTuple(encoder_state, encoder_state)
        encoder_state = (encoder_state,) * Params.decoder_num_layers
        tf.get_variable_scope().reuse_variables()

        with tf.variable_scope('EPD', reuse=tf.AUTO_REUSE):
            my_decoder = decoder.Model(new_arch_outputs, encoder_state, decoder_inputs, None,
                                       tf.estimator.ModeKeys.PREDICT, 'Decoder')
            new_sample_id = my_decoder.decode()

        tf.logging.info('Starting Session')
        config = tf.ConfigProto(allow_soft_placement=True)
        new_sample_id_list = []
        with tf.train.SingularMonitoredSession(
                config=config, checkpoint_dir=Params.autoencoder_model_dir) as sess:
            new_sample_id_v = sess.run(new_sample_id)
            new_sample_id_list.extend(new_sample_id_v.tolist())
        return new_sample_id_list


def get_train_ops(encoder_outputs, predictor_train_target,
                  reuse=tf.AUTO_REUSE):
    with tf.variable_scope('EPD', reuse=reuse):
        predictor = encoder.Predictor(encoder_outputs, predictor_train_target, tf.estimator.ModeKeys.TRAIN)
        predictor.build()
        predictor.compute_loss()

        tf.summary.scalar('training_loss', predictor.total_loss)

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
            gradients, variables = zip(*opt.compute_gradients(predictor.total_loss))
            grad_norm = tf.global_norm(gradients)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, Params.max_gradient_norm)
            train_op = opt.apply_gradients(
                zip(clipped_gradients, variables), global_step=global_step)

        return predictor.total_loss, learning_rate, train_op, global_step, grad_norm


def get_predict_ops(encoder_outputs, reuse=tf.AUTO_REUSE):
    with tf.variable_scope('EPD', reuse=reuse):
        def preprocess(encoder_src):
            return encoder_src, tf.constant([SOS], dtype=tf.int32)
        encoder_outputs = tf.convert_to_tensor(encoder_outputs, dtype=tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices(encoder_outputs)
        dataset = dataset.map(preprocess)
        dataset = dataset.batch(Params.controller_batch_size)
        iterator = dataset.make_one_shot_iterator()
        encoder_outputs, decoder_inputs = iterator.get_next()
        predictor = encoder.Predictor(encoder_outputs,
                                      None,
                                      tf.estimator.ModeKeys.PREDICT)
        predictor.build()
        predict_value, _, new_arch_outputs = predictor.infer()
        return predict_value, new_arch_outputs, decoder_inputs


def input_fn(encoder_input, predictor_target, batch_size):
    # The encoder_input and predictor_target in the parameters are converted to tf tensors and wrapped into datasets
    # ready to be input into tf models.
    # The encoder_input are the representation of architectures and are embedded before wrap.
    _, encoder_outputs = encode(encoder_input)
    encoder_outputs = tf.convert_to_tensor(encoder_outputs, dtype=tf.float32)
    encoder_outputs = tf.data.Dataset.from_tensor_slices(encoder_outputs)
    predictor_target = tf.convert_to_tensor(predictor_target, dtype=tf.float32)
    predictor_target = tf.data.Dataset.from_tensor_slices(predictor_target)
    dataset = tf.data.Dataset.zip((encoder_outputs, predictor_target))
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    encoder_outputs, predictor_target = iterator.get_next()
    return encoder_outputs, predictor_target


def input_fn_predict(original_encoder_input, batch_size):
    _, encoder_outputs = encode(original_encoder_input)
    return encoder_outputs


# encoder target is the list of performances of the architectures
def train(encoder_input, predictor_target):
    with tf.Graph().as_default():
        tf.logging.info('Training Encoder-Predictor-Decoder')
        tf.logging.info('Preparing data')
        if not isinstance(predictor_target, np.ndarray):
            predictor_target = np.array(predictor_target)
        if len(predictor_target.shape) == 1:
            predictor_target = predictor_target.reshape(predictor_target.shape[0], 1)
        encoder_outputs, predictor_train_target = input_fn(encoder_input,
                                                           predictor_target,
                                                           Params.controller_batch_size)
        tf.logging.info('Building model')
        train_loss, learning_rate, train_op, global_step, grad_norm = get_train_ops(encoder_outputs,
                                                                                    predictor_train_target)
        saver = tf.train.Saver(max_to_keep=10)
        checkpoint_saver_hook = tf.train.CheckpointSaverHook(
            Params.get_controller_model_dir(), save_steps=Params.batches_per_epoch * Params.save_frequency, saver=saver)
        hooks = [checkpoint_saver_hook]
        merged_summary = tf.summary.merge_all()
        tf.logging.info('Starting Session')
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.train.SingularMonitoredSession(
                config=config, hooks=hooks, checkpoint_dir=Params.get_controller_model_dir()) as sess:
            writer = tf.summary.FileWriter(Params.get_controller_model_dir(), sess.graph)
            start_time = time.time()
            for step in range(Params.train_epochs * Params.batches_per_epoch):
                run_ops = [
                    train_loss,
                    learning_rate,
                    train_op,
                    global_step,
                    grad_norm,
                    merged_summary,
                ]
                train_loss_v, learning_rate_v, _, global_step_v, gn_v, summary = sess.run(
                    run_ops)

                writer.add_summary(summary, global_step_v)

                epoch = (global_step_v + 1) // Params.batches_per_epoch

                curr_time = time.time()
                if (global_step_v + 1) % 100 == 0:
                    log_string = "epoch={:<6d} ".format(epoch)
                    log_string += "step={:<6d} ".format(global_step_v + 1)
                    log_string += "loss={:<6f} ".format(train_loss_v)
                    log_string += "learning_rate={:<8.4f} ".format(learning_rate_v)
                    log_string += "|gn|={:<8.4f} ".format(gn_v)
                    log_string += "mins={:<10.2f}".format((curr_time - start_time) / 60)
                    tf.logging.info(log_string)


def predict(encoder_input):
    with tf.Graph().as_default():
        tf.logging.info(
            'Generating new architectures using gradient descent with step size {}'.format(Params.predict_lambda))
        tf.logging.info('Preparing data')
        N = len(encoder_input)
        encoder_outputs = input_fn_predict(encoder_input,
                                           Params.controller_batch_size)
        predict_value, new_arch_outputs, decoder_inputs = get_predict_ops(encoder_outputs)
        tf.logging.info('Starting Session')
        config = tf.ConfigProto(allow_soft_placement=True)
        new_sample_id_list = []
        with tf.train.SingularMonitoredSession(
                config=config, checkpoint_dir=Params.get_controller_model_dir()) as sess:
            for _ in range(N // Params.controller_batch_size):
                new_sample_id_v = sess.run(new_arch_outputs)
                new_sample_id_list.extend(new_sample_id_v.tolist())

        new_sample_id_list = decode(decoder_inputs, np.array(new_sample_id_list))
        return new_sample_id_list


def main(unused_argv):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    num = 300

    # all_params = vars(Params)
    # with open(os.path.join(Params.output_dir, 'hparams.json'), 'w') as f:
    #     json.dump(all_params.__dict__, f)
    arch_pool = utils.generate_arch(num, 5, 5)
    predictor_target = np.array([0.5] * num).reshape(num)
    branch_length = 40 // 2 // 5 // 2
    encoder_input = list(
        map(lambda x: utils.parse_arch_to_seq(x[0], branch_length) + utils.parse_arch_to_seq(x[1], branch_length),
            arch_pool))
    Params.batches_per_epoch = math.ceil(len(encoder_input) / Params.controller_batch_size)
    train(encoder_input, predictor_target)
    result = predict(encoder_input)
    result = np.array(result)
    print(result.shape)


if __name__ == '__main__':
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
    tf.logging.set_verbosity(tf.logging.INFO)
    unparsed = set_params()
    tf.app.run(argv=[sys.argv[0]] + unparsed)
