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
        predictor = encoder.Predictor(embedding, predictor_train_target, params, tf.estimator.ModeKeys.TRAIN)
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


def decode(new_arch_emb, new_arch_outputs, decoder_input):
    return None, None


def get_predict_ops(embedding, decoder_input, params, reuse=tf.AUTO_REUSE):
    encoder_predict_target = None
    decoder_predict_target = None
    with tf.variable_scope('EPD', reuse=reuse):
        predictor = encoder.Predictor(embedding,
                                      None,
                                      params,
                                      tf.estimator.ModeKeys.PREDICT)
        _, predict_value, new_arch_emb, new_arch_outputs = predictor.infer()
        sample_id, new_sample_id = decode(new_arch_emb, new_arch_outputs, decoder_input)
        return predict_value, sample_id, new_sample_id


def input_fn(encoder_input, predictor_target, batch_size):
    return None, None


# encoder target is the list of performances of the architectures
def train(params, encoder_input, predictor_target):
    with tf.Graph().as_default():
        tf.logging.info('Training Encoder-Predictor-Decoder')
        tf.logging.info('Preparing data')
        embedding, predictor_train_target = input_fn(encoder_input,
                                                     predictor_target,
                                                     params['batch_size'])
        tf.logging.info('Building model')
        train_loss, learning_rate, train_op, global_step, grad_norm = get_train_ops(embedding,
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
        embedding, decoder_input = input_fn(encoder_input,
                                            None,
                                            params['batch_size'])
        predict_value, sample_id, new_sample_id = get_predict_ops(embedding, decoder_input, params)
        tf.logging.info('Starting Session')
        config = tf.ConfigProto(allow_soft_placement=True)
        new_sample_id_list = []
        with tf.train.SingularMonitoredSession(
                config=config, checkpoint_dir=params['model_dir']) as sess:
            for _ in range(N // params['batch_size']):
                new_sample_id_v = sess.run(new_sample_id)
                new_sample_id_list.extend(new_sample_id_v.tolist())
        return new_sample_id_list
