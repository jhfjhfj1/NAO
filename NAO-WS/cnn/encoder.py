from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import tensorflow as tf

from params import Params

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-5


class Predictor:
    def __init__(self, encoder_outputs, target, mode):
        self.encoder_outputs = encoder_outputs
        self.target = target
        self.emb_size = Params.encoder_emb_size
        self.mlp_num_layers = Params.mlp_num_layers
        self.mlp_hidden_size = Params.mlp_hidden_size
        self.mlp_dropout = Params.mlp_dropout
        self.dropout = Params.encoder_dropout
        self.output = None
        self.loss = None
        self.weight_decay = Params.weight_decay
        self.mode = mode
        self.time_major = Params.time_major
        self.prediction = None
        self.arch_emb = None

    def build(self):
        x = self.encoder_outputs
        if self.time_major:
            x = tf.transpose(x, [1, 0, 2])
        if self.time_major:
            x = tf.reduce_mean(x, axis=0)
        else:
            x = tf.reduce_mean(x, axis=1)
        x = tf.nn.l2_normalize(x, dim=-1)
        self.arch_emb = x
        for i in range(self.mlp_num_layers):
            name = 'mlp_{}'.format(i)
            x = tf.layers.dense(x, self.mlp_hidden_size, activation=tf.nn.relu, name=name)
            x = tf.layers.dropout(x, self.mlp_dropout)
        self.prediction = tf.layers.dense(x, 1, activation=tf.sigmoid, name='regression')

    def compute_loss(self):
        weights = 1 - tf.cast(tf.equal(self.target, -1.0), tf.float32)
        mean_squared_error = tf.losses.mean_squared_error(
            labels=self.target,
            predictions=self.prediction,
            weights=weights)
        tf.identity(mean_squared_error, name='squared_error')
        tf.summary.scalar('mean_squared_error', mean_squared_error)
        # Add weight decay to the loss.
        self.loss = mean_squared_error
        total_loss = mean_squared_error + self.weight_decay * tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        self.total_loss = total_loss

    def infer(self):
        assert self.mode == tf.estimator.ModeKeys.PREDICT
        grads_on_outputs = tf.gradients(self.prediction, self.encoder_outputs)[0]
        # lambdas = tf.expand_dims(tf.expand_dims(lambdas, axis=-1), axis=-1)
        new_arch_outputs = self.encoder_outputs - Params.predict_lambda * grads_on_outputs
        new_arch_outputs = tf.nn.l2_normalize(new_arch_outputs, dim=-1)
        if self.time_major:
            new_arch_emb = tf.reduce_mean(new_arch_outputs, axis=0)
        else:
            new_arch_emb = tf.reduce_mean(new_arch_outputs, axis=1)
        new_arch_emb = tf.nn.l2_normalize(new_arch_emb, dim=-1)
        return self.prediction, new_arch_emb, new_arch_outputs


class Encoder(object):
    def __init__(self, x, mode, scope='Encoder', reuse=tf.AUTO_REUSE):
        self.x = x
        self.batch_size = tf.shape(x)[0]
        self.vocab_size = Params.encoder_vocab_size
        self.emb_size = Params.encoder_emb_size
        self.hidden_size = Params.encoder_hidden_size
        self.encoder_length = Params.encoder_length
        self.weight_decay = Params.weight_decay
        self.mode = mode
        self.time_major = Params.time_major
        self.is_training = self.mode == tf.estimator.ModeKeys.TRAIN
        if not self.is_training:
            Params.encoder_dropout = 0.0
            Params.mlp_dropout = 0.0

        # initializer = tf.orthogonal_initializer()
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        tf.get_variable_scope().set_initializer(initializer)
        self.build_graph(scope=scope, reuse=reuse)

    def build_graph(self, scope=None, reuse=tf.AUTO_REUSE):
        tf.logging.info("# creating %s graph ..." % self.mode)
        # Encoder
        with tf.variable_scope(scope, reuse=reuse):
            self.W_emb = tf.get_variable('W_emb', [self.vocab_size, self.emb_size])
            self.arch_emb, self.encoder_outputs, self.encoder_state = self.build_encoder()
            if self.mode != tf.estimator.ModeKeys.PREDICT:
                self.compute_loss()
            else:
                self.loss = None
                self.total_loss = None

    def build_encoder(self):
        #encoder init
        self.num_layers = Params.encoder_num_layers
        self.hidden_size = Params.encoder_hidden_size
        self.emb_size = Params.encoder_emb_size
        self.mlp_num_layers = Params.mlp_num_layers
        self.mlp_hidden_size = Params.mlp_hidden_size
        self.mlp_dropout = Params.mlp_dropout
        self.source_length = Params.source_length
        self.encoder_length = Params.encoder_length
        self.vocab_size = Params.encoder_vocab_size
        self.dropout = Params.encoder_dropout
        self.time_major = Params.time_major

        x = self.x
        batch_size = self.batch_size

        self.batch_size = batch_size
        assert x.shape.ndims == 2, '[batch_size, length]'
        x = tf.gather(self.W_emb, x)
        if self.source_length != self.encoder_length:
            tf.logging.info('Concacting source sequence along depth')
            assert self.source_length % self.encoder_length == 0
            ratio = self.source_length // self.encoder_length
            x = tf.reshape(x, [batch_size, self.source_length // ratio, ratio * self.emb_size])
        if self.time_major:
            x = tf.transpose(x, [1, 0, 2])
        cell_list = []
        for i in range(self.num_layers):
            lstm_cell = tf.contrib.rnn.LSTMCell(
                self.hidden_size)
            lstm_cell = tf.contrib.rnn.DropoutWrapper(
                lstm_cell,
                output_keep_prob=1 - self.dropout)
            cell_list.append(lstm_cell)
        if len(cell_list) == 1:
            cell = cell_list[0]
        else:
            cell = tf.contrib.rnn.MultiRNNCell(cell_list)
        initial_state = cell.zero_state(batch_size, dtype=tf.float32)
        x, state = tf.nn.dynamic_rnn(cell,
                                     x,
                                     dtype=tf.float32,
                                     time_major=self.time_major,
                                     initial_state=initial_state)
        x = tf.nn.l2_normalize(x, dim=-1)
        self.encoder_outputs = x
        self.encoder_state = state

        if self.time_major:
            x = tf.reduce_mean(x, axis=0)
        else:
            x = tf.reduce_mean(x, axis=1)
        x = tf.nn.l2_normalize(x, dim=-1)

        self.arch_emb = x

        '''
        x = tf.layers.batch_normalization(
        x, axis=1,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
        center=True, scale=True, training=is_training, fused=True
        )'''
        res = {
            'arch_emb': self.arch_emb,
            'encoder_outputs': self.encoder_outputs,
            'encoder_state': self.encoder_state,
        }

        return res['arch_emb'], res['encoder_outputs'], res['encoder_state']

    def compute_loss(self):
        total_loss = self.weight_decay * tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        self.total_loss = total_loss

    def train(self):
        assert self.mode == tf.estimator.ModeKeys.TRAIN
        self.global_step = tf.train.get_or_create_global_step()
        self.learning_rate = tf.constant(Params.controller_lr)
        if Params.optimizer == "sgd":
            self.learning_rate = tf.cond(
                self.global_step < Params.start_decay_step,
                lambda: self.learning_rate,
                lambda: tf.train.exponential_decay(
                    self.learning_rate,
                    (self.global_step - Params.start_decay_step),
                    Params.decay_steps,
                    Params.decay_factor,
                    staircase=True),
                name="learning_rate")
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif Params.optimizer == "adam":
            assert float(
                Params.controller_lr
            ) <= 0.001, "! High Adam learning rate %g" % Params.controller_lr
            opt = tf.train.AdamOptimizer(self.learning_rate)
        elif Params.optimizer == 'adadelta':
            opt = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            gradients, variables = zip(*opt.compute_gradients(self.total_loss))
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, Params.max_gradient_norm)
            self.train_op = opt.apply_gradients(
                zip(clipped_gradients, variables), global_step=self.global_step)

        tf.identity(self.learning_rate, 'learning_rate')
        tf.summary.scalar("learning_rate", self.learning_rate),
        tf.summary.scalar("total_loss", self.total_loss),

        return {
            'train_op': self.train_op,
            'loss': self.total_loss,
        }

    def eval(self):
        assert self.mode == tf.estimator.ModeKeys.EVAL
        return {
            'loss': self.total_loss,
        }

    def predict(self):
        assert self.mode == tf.estimator.ModeKeys.PREDICT
        # lambdas = tf.expand_dims(tf.expand_dims(lambdas, axis=-1), axis=-1)
        return self.arch_emb, self.encoder_outputs

