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

SOS = 0
EOS = 0


def encode(params, original_encoder_input):
    N = np.array(original_encoder_input).shape[0]
    with tf.Graph().as_default():
        tf.logging.info(
            'Generating new architectures using gradient descent with step size {}'.format(params['predict_lambda']))
        tf.logging.info('Preparing data')
        encoder_input = tf.convert_to_tensor(original_encoder_input, dtype=tf.int32)
        encoder_input = tf.data.Dataset.from_tensor_slices(encoder_input)
        encoder_input = encoder_input.batch(params['batch_size'])
        iterator = encoder_input.make_one_shot_iterator()
        encoder_input = iterator.get_next()

        with tf.variable_scope('EPD', reuse=tf.AUTO_REUSE):
            my_encoder = encoder.Encoder(encoder_input, params, tf.estimator.ModeKeys.PREDICT, 'Encoder', tf.AUTO_REUSE)
            embed = my_encoder.predict()

        tf.logging.info('Starting Session')
        config = tf.ConfigProto(allow_soft_placement=True)
        arch_embed, encoder_outputs = [], []
        with tf.train.SingularMonitoredSession(
                config=config, checkpoint_dir=params['autoencoder_model_dir']) as sess:
            for _ in range(N // params['batch_size']):
                a, b = sess.run(embed)
                arch_embed.append(a)
                encoder_outputs.append(b)
        arch_embed = np.array(arch_embed)
        encoder_outputs = np.array(encoder_outputs)
        encoder_outputs = encoder_outputs.transpose([0, 2, 1, 3])
        return arch_embed.reshape(
            (arch_embed.shape[0] * arch_embed.shape[1],) + arch_embed.shape[2:]), encoder_outputs.reshape(
            (encoder_outputs.shape[0] * encoder_outputs.shape[1],) + encoder_outputs.shape[2:])


def decode(params, decoder_inputs, new_arch_outputs):
    if params['time_major']:
        new_arch_outputs = tf.transpose(new_arch_outputs, [1, 0, 2])
    new_arch_outputs = tf.nn.l2_normalize(new_arch_outputs, dim=-1)
    if params['time_major']:
        new_arch_emb = tf.reduce_mean(new_arch_outputs, axis=0)
    else:
        new_arch_emb = tf.reduce_mean(new_arch_outputs, axis=1)
    new_arch_emb = tf.nn.l2_normalize(new_arch_emb, dim=-1)

    encoder_state = new_arch_emb
    encoder_state.set_shape([None, params['decoder_hidden_size']])
    encoder_state = tf.contrib.rnn.LSTMStateTuple(encoder_state, encoder_state)
    encoder_state = (encoder_state,) * params['decoder_num_layers']
    tf.get_variable_scope().reuse_variables()

    my_decoder = decoder.Model(new_arch_outputs, encoder_state, decoder_inputs, None,
                               params, tf.estimator.ModeKeys.PREDICT, 'Decoder')
    new_sample_id = my_decoder.decode()

    tf.logging.info('Starting Session')
    config = tf.ConfigProto(allow_soft_placement=True)
    new_sample_id_list = []
    with tf.train.SingularMonitoredSession(
            config=config, checkpoint_dir=params['model_dir']) as sess:
        for _ in range(len(new_arch_outputs) // params['batch_size']):
            new_sample_id_v = sess.run(new_sample_id)
            new_sample_id_list.extend(new_sample_id_v.tolist())
    return new_sample_id_list


def get_train_ops(encoder_outputs, predictor_train_target, params,
                  reuse=tf.AUTO_REUSE):
    with tf.variable_scope('EPD', reuse=reuse):
        predictor = encoder.Predictor(encoder_outputs, predictor_train_target, params, tf.estimator.ModeKeys.TRAIN)
        predictor.build()
        predictor.compute_loss()

        tf.summary.scalar('training_loss', predictor.total_loss)

        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.constant(params['lr'])
        if params['optimizer'] == "sgd":
            learning_rate = tf.cond(
                global_step < params['start_decay_step'],
                lambda: learning_rate,
                lambda: tf.train.exponential_decay(
                    learning_rate,
                    (global_step - params['start_decay_step']),
                    params['decay_steps'],
                    params['decay_factor'],
                    staircase=True),
                name="calc_learning_rate")
            opt = tf.train.GradientDescentOptimizer(learning_rate)
        elif params['optimizer'] == "adam":
            assert float(params['lr']) <= 0.001, "! High Adam learning rate %g" % params['lr']
            opt = tf.train.AdamOptimizer(learning_rate)
        elif params['optimizer'] == 'adadelta':
            opt = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
        tf.summary.scalar("learning_rate", learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            gradients, variables = zip(*opt.compute_gradients(predictor.total_loss))
            grad_norm = tf.global_norm(gradients)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, params['max_gradient_norm'])
            train_op = opt.apply_gradients(
                zip(clipped_gradients, variables), global_step=global_step)

        return predictor.total_loss, learning_rate, train_op, global_step, grad_norm


def get_predict_ops(encoder_outputs, decoder_inputs, params, reuse=tf.AUTO_REUSE):
    with tf.variable_scope('EPD', reuse=reuse):
        predictor = encoder.Predictor(encoder_outputs,
                                      None,
                                      params,
                                      tf.estimator.ModeKeys.PREDICT)
        predictor.build()
        predict_value, new_arch_emb, new_arch_outputs = predictor.infer()
        new_sample_id = decode(params, decoder_inputs, new_arch_outputs)
        return predict_value, new_sample_id


def input_fn(encoder_input, predictor_target, batch_size, params):
    _, encoder_outputs = encode(params, encoder_input)
    encoder_outputs = tf.convert_to_tensor(encoder_outputs, dtype=tf.float32)
    encoder_outputs = tf.data.Dataset.from_tensor_slices(encoder_outputs)
    predictor_target = tf.convert_to_tensor(predictor_target, dtype=tf.float32)
    predictor_target = tf.data.Dataset.from_tensor_slices(predictor_target)
    dataset = tf.data.Dataset.zip((encoder_outputs, predictor_target))
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    encoder_outputs, predictor_target = iterator.get_next()
    return encoder_outputs, predictor_target


def input_fn_predict(original_encoder_input, batch_size, params):
    encoder_input = tf.convert_to_tensor(original_encoder_input, dtype=tf.int32)
    encoder_input = tf.data.Dataset.from_tensor_slices(encoder_input)

    def preprocess(encoder_src):
        return encoder_src, tf.constant([SOS], dtype=tf.int32)

    dataset = encoder_input.map(preprocess)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(10)
    iterator = dataset.make_one_shot_iterator()
    encoder_input, decoder_input = iterator.get_next()

    _, encoder_outputs = encode(params, original_encoder_input)
    encoder_outputs = tf.convert_to_tensor(encoder_outputs, dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices(encoder_outputs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    encoder_outputs = iterator.get_next()
    return encoder_outputs, decoder_input


# encoder target is the list of performances of the architectures
def train(params, encoder_input, predictor_target):
    with tf.Graph().as_default():
        tf.logging.info('Training Encoder-Predictor-Decoder')
        tf.logging.info('Preparing data')
        encoder_outputs, predictor_train_target = input_fn(encoder_input,
                                                           predictor_target,
                                                           params['batch_size'],
                                                           params)
        tf.logging.info('Building model')
        train_loss, learning_rate, train_op, global_step, grad_norm = get_train_ops(encoder_outputs,
                                                                                    predictor_train_target,
                                                                                    params)
        saver = tf.train.Saver(max_to_keep=10)
        checkpoint_saver_hook = tf.train.CheckpointSaverHook(
            params['model_dir'], save_steps=params['batches_per_epoch'] * params['save_frequency'], saver=saver)
        hooks = [checkpoint_saver_hook]
        merged_summary = tf.summary.merge_all()
        tf.logging.info('Starting Session')
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.train.SingularMonitoredSession(
                config=config, hooks=hooks, checkpoint_dir=params['model_dir']) as sess:
            writer = tf.summary.FileWriter(params['model_dir'], sess.graph)
            start_time = time.time()
            for step in range(params['train_epochs'] * params['batches_per_epoch']):
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

                epoch = (global_step_v + 1) // params['batches_per_epoch']

                curr_time = time.time()
                if (global_step_v + 1) % 100 == 0:
                    log_string = "epoch={:<6d} ".format(epoch)
                    log_string += "step={:<6d} ".format(global_step_v + 1)
                    log_string += "loss={:<6f} ".format(train_loss_v)
                    log_string += "learning_rate={:<8.4f} ".format(learning_rate_v)
                    log_string += "|gn|={:<8.4f} ".format(gn_v)
                    log_string += "mins={:<10.2f}".format((curr_time - start_time) / 60)
                    tf.logging.info(log_string)


def predict(params, encoder_input):
    with tf.Graph().as_default():
        tf.logging.info(
            'Generating new architectures using gradient descent with step size {}'.format(params['predict_lambda']))
        tf.logging.info('Preparing data')
        N = len(encoder_input)
        encoder_outputs, decoder_inputs = input_fn_predict(encoder_input,
                                                           params['batch_size'],
                                                           params)
        predict_value, new_sample_id = get_predict_ops(encoder_outputs, decoder_inputs, params)
        tf.logging.info('Starting Session')
        config = tf.ConfigProto(allow_soft_placement=True)
        new_sample_id_list = []
        with tf.train.SingularMonitoredSession(
                config=config, checkpoint_dir=params['model_dir']) as sess:
            for _ in range(N // params['batch_size']):
                new_sample_id_v = sess.run(new_sample_id)
                new_sample_id_list.extend(new_sample_id_v.tolist())
        return new_sample_id_list


def get_controller_params():
    params = {
        'model_dir': os.path.join(FLAGS.output_dir, 'controller'),
        'autoencoder_model_dir': os.path.join(FLAGS.output_dir, 'autoencoder'),
        'num_seed_arch': FLAGS.controller_num_seed_arch,
        'encoder_num_layers': FLAGS.controller_encoder_num_layers,
        'encoder_hidden_size': FLAGS.controller_encoder_hidden_size,
        'encoder_emb_size': FLAGS.controller_encoder_emb_size,
        'mlp_num_layers': FLAGS.controller_mlp_num_layers,
        'mlp_hidden_size': FLAGS.controller_mlp_hidden_size,
        'decoder_num_layers': FLAGS.controller_decoder_num_layers,
        'decoder_hidden_size': FLAGS.controller_decoder_hidden_size,
        'source_length': FLAGS.controller_source_length,
        'encoder_length': FLAGS.controller_encoder_length,
        'decoder_length': FLAGS.controller_decoder_length,
        'encoder_dropout': FLAGS.controller_encoder_dropout,
        'mlp_dropout': FLAGS.controller_mlp_dropout,
        'decoder_dropout': FLAGS.controller_decoder_dropout,
        'weight_decay': FLAGS.controller_weight_decay,
        'encoder_vocab_size': FLAGS.controller_encoder_vocab_size,
        'decoder_vocab_size': FLAGS.controller_decoder_vocab_size,
        'trade_off': FLAGS.controller_trade_off,
        'train_epochs': FLAGS.controller_train_epochs,
        'save_frequency': FLAGS.controller_save_frequency,
        'batch_size': FLAGS.controller_batch_size,
        'lr': FLAGS.controller_lr,
        'optimizer': FLAGS.controller_optimizer,
        'start_decay_step': FLAGS.controller_start_decay_step,
        'decay_steps': FLAGS.controller_decay_steps,
        'decay_factor': FLAGS.controller_decay_factor,
        'attention': FLAGS.controller_attention,
        'max_gradient_norm': FLAGS.controller_max_gradient_norm,
        'time_major': FLAGS.controller_time_major,
        'symmetry': FLAGS.controller_symmetry,
        'predict_beam_width': FLAGS.controller_predict_beam_width,
        'predict_lambda': FLAGS.controller_predict_lambda
    }
    return params


def main(unused_argv):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    num = 300

    all_params = vars(FLAGS)
    with open(os.path.join(FLAGS.output_dir, 'hparams.json'), 'w') as f:
        json.dump(all_params, f)
    arch_pool = utils.generate_arch(num, 5, 5)
    predictor_target = np.array([0.5] * num).reshape(num, 1)
    branch_length = 40 // 2 // 5 // 2
    encoder_input = list(
        map(lambda x: utils.parse_arch_to_seq(x[0], branch_length) + utils.parse_arch_to_seq(x[1], branch_length),
            arch_pool))
    controller_params = get_controller_params()
    controller_params['batches_per_epoch'] = math.ceil(len(encoder_input) / controller_params['batch_size'])
    train(controller_params, encoder_input, predictor_target)
    predict(controller_params, encoder_input)


parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--mode', type=str, default='train',
                    choices=['train', 'test'])

parser.add_argument('--data_path', type=str, default='/tmp/cifar10_data')

parser.add_argument('--eval_dataset', type=str, default='valid',
                    choices=['valid', 'test', 'both'])

parser.add_argument('--output_dir', type=str, default='models')

parser.add_argument('--child_sample_policy', type=str, default=None)

parser.add_argument('--child_batch_size', type=int, default=128)

parser.add_argument('--child_eval_batch_size', type=int, default=128)

parser.add_argument('--child_num_epochs', type=int, default=150)

parser.add_argument('--child_lr_dec_every', type=int, default=100)

parser.add_argument('--child_num_layers', type=int, default=5)

parser.add_argument('--child_num_cells', type=int, default=5)

parser.add_argument('--child_out_filters', type=int, default=20)

parser.add_argument('--child_out_filters_scale', type=int, default=1)

parser.add_argument('--child_num_branches', type=int, default=5)

parser.add_argument('--child_num_aggregate', type=int, default=None)

parser.add_argument('--child_num_replicas', type=int, default=None)

parser.add_argument('--child_lr_T_0', type=int, default=None)

parser.add_argument('--child_lr_T_mul', type=int, default=None)

parser.add_argument('--child_cutout_size', type=int, default=None)

parser.add_argument('--child_grad_bound', type=float, default=5.0)

parser.add_argument('--child_lr', type=float, default=0.1)

parser.add_argument('--child_lr_dec_rate', type=float, default=0.1)

parser.add_argument('--child_lr_max', type=float, default=None)

parser.add_argument('--child_lr_min', type=float, default=None)

parser.add_argument('--child_keep_prob', type=float, default=0.5)

parser.add_argument('--child_drop_path_keep_prob', type=float, default=1.0)

parser.add_argument('--child_l2_reg', type=float, default=1e-4)

parser.add_argument('--child_fixed_arc', type=str, default=None)

parser.add_argument('--child_use_aux_heads', action='store_true', default=False)

parser.add_argument('--child_sync_replicas', action='store_true', default=False)

parser.add_argument('--child_lr_cosine', action='store_true', default=False)

parser.add_argument('--child_eval_every_epochs', type=str, default='30')

parser.add_argument('--child_arch_pool', type=str, default=None)

parser.add_argument('--child_data_format', type=str, default="NHWC", choices=['NHWC', 'NCHW'])

parser.add_argument('--controller_num_seed_arch', type=int, default=1000)

parser.add_argument('--controller_encoder_num_layers', type=int, default=1)

parser.add_argument('--controller_encoder_hidden_size', type=int, default=96)

parser.add_argument('--controller_encoder_emb_size', type=int, default=32)

parser.add_argument('--controller_mlp_num_layers', type=int, default=0)

parser.add_argument('--controller_mlp_hidden_size', type=int, default=200)

parser.add_argument('--controller_decoder_num_layers', type=int, default=1)

parser.add_argument('--controller_decoder_hidden_size', type=int, default=96)

parser.add_argument('--controller_source_length', type=int, default=60)

parser.add_argument('--controller_encoder_length', type=int, default=20)

parser.add_argument('--controller_decoder_length', type=int, default=60)

parser.add_argument('--controller_encoder_dropout', type=float, default=0.1)

parser.add_argument('--controller_mlp_dropout', type=float, default=0.1)

parser.add_argument('--controller_decoder_dropout', type=float, default=0.0)

parser.add_argument('--controller_weight_decay', type=float, default=1e-4)

parser.add_argument('--controller_encoder_vocab_size', type=int, default=12)

parser.add_argument('--controller_decoder_vocab_size', type=int, default=12)

parser.add_argument('--controller_trade_off', type=float, default=0.8)

parser.add_argument('--controller_train_epochs', type=int, default=300)

parser.add_argument('--controller_save_frequency', type=int, default=10)

parser.add_argument('--controller_batch_size', type=int, default=100)

parser.add_argument('--controller_lr', type=float, default=0.001)

parser.add_argument('--controller_optimizer', type=str, default='adam')

parser.add_argument('--controller_start_decay_step', type=int, default=100)

parser.add_argument('--controller_decay_steps', type=int, default=1000)

parser.add_argument('--controller_decay_factor', type=float, default=0.9)

parser.add_argument('--controller_attention', action='store_true', default=False)

parser.add_argument('--controller_max_gradient_norm', type=float, default=5.0)

parser.add_argument('--controller_time_major', action='store_true', default=False)

parser.add_argument('--controller_symmetry', action='store_true', default=False)

parser.add_argument('--controller_predict_beam_width', type=int, default=0)

parser.add_argument('--controller_predict_lambda', type=float, default=1)

if __name__ == '__main__':
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(argv=[sys.argv[0]] + unparsed)
