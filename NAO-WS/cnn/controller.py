from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import encoder
import decoder
import time

SOS = 0
EOS = 0


def get_train_ops(embedding, predictor_train_target, params,
                  reuse=tf.AUTO_REUSE):
    with tf.variable_scope('EPD', reuse=reuse):
        predictor = encoder.Predictor(params)
        predictor.build(embedding)
        predictor.compute_loss(predictor_train_target)

        total_loss = predictor.loss + params[
            'weight_decay'] * tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

        tf.summary.scalar('training_loss', total_loss)

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
            gradients, variables = zip(*opt.compute_gradients(total_loss))
            grad_norm = tf.global_norm(gradients)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, params['max_gradient_norm'])
            train_op = opt.apply_gradients(
                zip(clipped_gradients, variables), global_step=global_step)

        return total_loss, learning_rate, train_op, global_step, grad_norm


def get_predict_ops(encoder_predict_input, decoder_predict_input, params, reuse=tf.AUTO_REUSE):
    encoder_predict_target = None
    decoder_predict_target = None
    with tf.variable_scope('EPD', reuse=reuse):
        my_encoder = encoder.Model(encoder_predict_input, encoder_predict_target, params, tf.estimator.ModeKeys.PREDICT,
                                   'Encoder', reuse)
        encoder_outputs = my_encoder.encoder_outputs
        encoder_state = my_encoder.arch_emb
        encoder_state.set_shape([None, params['decoder_hidden_size']])
        encoder_state = tf.contrib.rnn.LSTMStateTuple(encoder_state, encoder_state)
        encoder_state = (encoder_state,) * params['decoder_num_layers']
        my_decoder = decoder.Model(encoder_outputs, encoder_state, decoder_predict_input, decoder_predict_target,
                                   params, tf.estimator.ModeKeys.PREDICT, 'Decoder', reuse)
        arch_emb, predict_value, new_arch_emb, new_arch_outputs = my_encoder.infer()
        # the sample_id is not used by anything.
        sample_id = my_decoder.decode()

        encoder_state = new_arch_emb
        encoder_state.set_shape([None, params['decoder_hidden_size']])
        encoder_state = tf.contrib.rnn.LSTMStateTuple(encoder_state, encoder_state)
        encoder_state = (encoder_state,) * params['decoder_num_layers']
        tf.get_variable_scope().reuse_variables()
        my_decoder = decoder.Model(new_arch_outputs, encoder_state, decoder_predict_input, decoder_predict_target,
                                   params, tf.estimator.ModeKeys.PREDICT, 'Decoder')
        new_sample_id = my_decoder.decode()

        return predict_value, sample_id, new_sample_id


def input_fn(embedding, predictor_target, batch_size):
    return None, None


# encoder target is the list of performances of the architectures
def train(params, embedding, predictor_target):
    with tf.Graph().as_default():
        tf.logging.info('Training Encoder-Predictor-Decoder')
        tf.logging.info('Preparing data')
        embedding, predictor_train_target = input_fn(
            embedding,
            predictor_target,
            params['batch_size'],
        )
        tf.logging.info('Building model')
        train_mse, train_cross_entropy, train_loss, learning_rate, train_op, global_step, grad_norm = get_train_ops(
            encoder_train_input, predictor_train_target, decoder_train_input, decoder_train_target, params)
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
                    train_mse,
                    train_cross_entropy,
                    train_loss,
                    learning_rate,
                    train_op,
                    global_step,
                    grad_norm,
                    merged_summary,
                ]
                train_mse_v, train_cross_entropy_v, train_loss_v, learning_rate_v, _, global_step_v, gn_v, summary = sess.run(
                    run_ops)

                writer.add_summary(summary, global_step_v)

                epoch = (global_step_v + 1) // params['batches_per_epoch']

                curr_time = time.time()
                if (global_step_v + 1) % 100 == 0:
                    log_string = "epoch={:<6d} ".format(epoch)
                    log_string += "step={:<6d} ".format(global_step_v + 1)
                    log_string += "se={:<6f} ".format(train_mse_v)
                    log_string += "cross_entropy={:<6f} ".format(train_cross_entropy_v)
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
        encoder_input, decoder_input = input_fn(
            encoder_input,
            None,
            None,
            'test',
            params['batch_size'],
            1,
            False,
        )
        predict_value, sample_id, new_sample_id = get_predict_ops(encoder_input, decoder_input, params)
        tf.logging.info('Starting Session')
        config = tf.ConfigProto(allow_soft_placement=True)
        new_sample_id_list = []
        with tf.train.SingularMonitoredSession(
                config=config, checkpoint_dir=params['model_dir']) as sess:
            for _ in range(N // params['batch_size']):
                new_sample_id_v = sess.run(new_sample_id)
                new_sample_id_list.extend(new_sample_id_v.tolist())
        return new_sample_id_list
