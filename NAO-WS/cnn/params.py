import argparse

import os
import tensorflow as tf

from utils import build_dag


class Params:
    augment = True
    dataset = 'cifar10'
    base_dir = None
    arch_pool_prob = None
    batches_per_epoch = None
    pass_hidden_state = None
    history_dir = 'models'
    output_dir = None
    autoencoder_model_dir = None
    num_seed_arch = None
    encoder_num_layers = None
    encoder_hidden_size = None
    encoder_emb_size = None
    mlp_num_layers = None
    mlp_hidden_size = None
    decoder_num_layers = None
    decoder_hidden_size = None
    source_length = None
    encoder_length = None
    decoder_length = None
    encoder_dropout = None
    mlp_dropout = None
    decoder_dropout = None
    weight_decay = None
    encoder_vocab_size = None
    decoder_vocab_size = None
    trade_off = None
    train_epochs = None
    save_frequency = None
    controller_batch_size = None
    controller_lr = None
    optimizer = None
    start_decay_step = None
    decay_steps = None
    decay_factor = None
    attention = None
    max_gradient_norm = None
    time_major = None
    symmetry = None
    predict_beam_width = None
    predict_lambda = None

    # child_params
    data_dir = None
    sample_policy = None
    child_batch_size = None
    eval_batch_size = None
    num_epochs = None
    lr_dec_every = None
    num_layers = None
    num_cells = None
    out_filters = None
    out_filters_scale = None
    num_aggregate = None
    num_replicas = None
    lr_T_0 = None
    lr_T_mul = None
    cutout_size = None
    grad_bound = None
    lr_dec_rate = None
    lr_max = None
    lr_min = None
    drop_path_keep_prob = None
    keep_prob = None
    l2_reg = None
    fixed_arc = None
    use_aux_heads = None
    sync_replicas = None
    lr_cosine = None
    eval_every_epochs = None
    data_format = None
    child_lr = None
    arch_pool = None

    @classmethod
    def get_controller_model_dir(cls):
        return os.path.join(cls.output_dir, cls.dataset, 'controller')

    @classmethod
    def get_child_model_dir(cls):
        return os.path.join(cls.output_dir, cls.dataset, 'child')

    @classmethod
    def set_params(cls, flags):
        # controller_params
        cls.base_dir = flags.base_dir
        cls.output_dir = flags.output_dir
        cls.autoencoder_model_dir = os.path.join(cls.base_dir, 'autoencoder')
        cls.num_seed_arch = flags.controller_num_seed_arch
        cls.encoder_num_layers = flags.controller_encoder_num_layers
        cls.encoder_hidden_size = flags.controller_encoder_hidden_size
        cls.encoder_emb_size = flags.controller_encoder_emb_size
        cls.mlp_num_layers = flags.controller_mlp_num_layers
        cls.mlp_hidden_size = flags.controller_mlp_hidden_size
        cls.decoder_num_layers = flags.controller_decoder_num_layers
        cls.decoder_hidden_size = flags.controller_decoder_hidden_size
        cls.source_length = flags.controller_source_length
        cls.encoder_length = flags.controller_encoder_length
        cls.decoder_length = flags.controller_decoder_length
        cls.encoder_dropout = flags.controller_encoder_dropout
        cls.mlp_dropout = flags.controller_mlp_dropout
        cls.decoder_dropout = flags.controller_decoder_dropout
        cls.weight_decay = flags.controller_weight_decay
        cls.encoder_vocab_size = flags.controller_encoder_vocab_size
        cls.decoder_vocab_size = flags.controller_decoder_vocab_size
        cls.trade_off = flags.controller_trade_off
        cls.train_epochs = flags.controller_train_epochs
        cls.save_frequency = flags.controller_save_frequency
        cls.controller_batch_size = flags.controller_batch_size
        cls.controller_lr = flags.controller_lr
        cls.optimizer = flags.controller_optimizer
        cls.start_decay_step = flags.controller_start_decay_step
        cls.decay_steps = flags.controller_decay_steps
        cls.decay_factor = flags.controller_decay_factor
        cls.attention = flags.controller_attention
        cls.max_gradient_norm = flags.controller_max_gradient_norm
        cls.time_major = flags.controller_time_major
        cls.symmetry = flags.controller_symmetry
        cls.predict_beam_width = flags.controller_predict_beam_width
        cls.predict_lambda = flags.controller_predict_lambda

        # child_params
        cls.sample_policy = flags.child_sample_policy
        cls.child_batch_size = flags.child_batch_size
        cls.eval_batch_size = flags.child_eval_batch_size
        cls.num_epochs = flags.child_num_epochs
        cls.lr_dec_every = flags.child_lr_dec_every
        cls.num_layers = flags.child_num_layers
        cls.num_cells = flags.child_num_cells
        cls.out_filters = flags.child_out_filters
        cls.out_filters_scale = flags.child_out_filters_scale
        cls.num_aggregate = flags.child_num_aggregate
        cls.num_replicas = flags.child_num_replicas
        cls.lr_T_0 = flags.child_lr_T_0
        cls.lr_T_mul = flags.child_lr_T_mul
        cls.cutout_size = flags.child_cutout_size
        cls.grad_bound = flags.child_grad_bound
        cls.lr_dec_rate = flags.child_lr_dec_rate
        cls.lr_max = flags.child_lr_max
        cls.lr_min = flags.child_lr_min
        cls.drop_path_keep_prob = flags.child_drop_path_keep_prob
        cls.keep_prob = flags.child_keep_prob
        cls.l2_reg = flags.child_l2_reg
        cls.fixed_arc = flags.child_fixed_arc
        cls.use_aux_heads = flags.child_use_aux_heads
        cls.sync_replicas = flags.child_sync_replicas
        cls.lr_cosine = flags.child_lr_cosine
        cls.eval_every_epochs = eval(flags.child_eval_every_epochs)
        cls.data_format = flags.child_data_format
        cls.child_lr = flags.child_lr
        cls.arch_pool = None

        if flags.child_arch_pool is not None:
            with open(flags.child_arch_pool) as f:
                archs = f.read().splitlines()
                archs = list(map(build_dag, archs))
                cls.arch_pool = archs
        if os.path.exists(os.path.join(cls.get_child_model_dir(), 'arch_pool')):
            tf.logging.info('Found arch_pool in child model dir, loading')
            with open(os.path.join(cls.get_child_model_dir(), 'arch_pool')) as f:
                archs = f.read().splitlines()
                archs = list(map(build_dag, archs))
                cls.arch_pool = archs

        cls.dataset = flags.dataset
        cls.augment = flags.augment


def construct_parser():
    global parser
    parser = argparse.ArgumentParser()
    # Basic model parameters.
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'])
    parser.add_argument('--eval_dataset', type=str, default='valid',
                        choices=['valid', 'test', 'both'])
    parser.add_argument('--output_dir', type=str, default='models')
    parser.add_argument('--base_dir', type=str, default='./')
    parser.add_argument('--child_sample_policy', type=str, default='uniform')
    parser.add_argument('--child_batch_size', type=int, default=160)
    parser.add_argument('--child_eval_batch_size', type=int, default=500)
    parser.add_argument('--child_num_epochs', type=int, default=150)
    parser.add_argument('--child_lr_dec_every', type=int, default=100)
    parser.add_argument('--child_num_layers', type=int, default=6)
    parser.add_argument('--child_num_cells', type=int, default=5)
    parser.add_argument('--child_out_filters', type=int, default=20)
    parser.add_argument('--child_out_filters_scale', type=int, default=1)
    parser.add_argument('--child_num_branches', type=int, default=5)
    parser.add_argument('--child_num_aggregate', type=int, default=None)
    parser.add_argument('--child_num_replicas', type=int, default=None)
    parser.add_argument('--child_lr_T_0', type=int, default=10)
    parser.add_argument('--child_lr_T_mul', type=int, default=2)
    parser.add_argument('--child_cutout_size', type=int, default=None)
    parser.add_argument('--child_grad_bound', type=float, default=5.0)
    parser.add_argument('--child_lr', type=float, default=0.1)
    parser.add_argument('--child_lr_dec_rate', type=float, default=0.1)
    parser.add_argument('--child_lr_max', type=float, default=0.05)
    parser.add_argument('--child_lr_min', type=float, default=0.0005)
    parser.add_argument('--child_keep_prob', type=float, default=0.90)
    parser.add_argument('--child_drop_path_keep_prob', type=float, default=0.60)
    parser.add_argument('--child_l2_reg', type=float, default=1e-4)
    parser.add_argument('--child_fixed_arc', type=str, default=None)
    parser.add_argument('--child_use_aux_heads', action='store_true', default=True)
    parser.add_argument('--child_sync_replicas', action='store_true', default=False)
    parser.add_argument('--child_lr_cosine', action='store_true', default=True)
    parser.add_argument('--child_eval_every_epochs', type=str, default='30')
    parser.add_argument('--child_arch_pool', type=str, default=None)
    parser.add_argument('--child_data_format', type=str, default="NHWC", choices=['NHWC', 'NCHW'])
    parser.add_argument('--controller_num_seed_arch', type=int, default=20)
    parser.add_argument('--controller_encoder_num_layers', type=int, default=1)
    parser.add_argument('--controller_encoder_hidden_size', type=int, default=96)
    parser.add_argument('--controller_encoder_emb_size', type=int, default=48)
    parser.add_argument('--controller_mlp_num_layers', type=int, default=3)
    parser.add_argument('--controller_mlp_hidden_size', type=int, default=100)
    parser.add_argument('--controller_decoder_num_layers', type=int, default=1)
    parser.add_argument('--controller_decoder_hidden_size', type=int, default=96)
    parser.add_argument('--controller_source_length', type=int, default=40)
    parser.add_argument('--controller_encoder_length', type=int, default=20)
    parser.add_argument('--controller_decoder_length', type=int, default=40)
    parser.add_argument('--controller_encoder_dropout', type=float, default=0.1)
    parser.add_argument('--controller_mlp_dropout', type=float, default=0.1)
    parser.add_argument('--controller_decoder_dropout', type=float, default=0.0)
    parser.add_argument('--controller_weight_decay', type=float, default=1e-4)
    parser.add_argument('--controller_encoder_vocab_size', type=int, default=12)
    parser.add_argument('--controller_decoder_vocab_size', type=int, default=12)
    parser.add_argument('--controller_trade_off', type=float, default=0.8)
    parser.add_argument('--controller_train_epochs', type=int, default=1000)
    parser.add_argument('--controller_save_frequency', type=int, default=100)
    parser.add_argument('--controller_batch_size', type=int, default=100)
    parser.add_argument('--controller_lr', type=float, default=0.001)
    parser.add_argument('--controller_optimizer', type=str, default='adam')
    parser.add_argument('--controller_start_decay_step', type=int, default=100)
    parser.add_argument('--controller_decay_steps', type=int, default=1000)
    parser.add_argument('--controller_decay_factor', type=float, default=0.9)
    parser.add_argument('--controller_attention', action='store_true', default=True)
    parser.add_argument('--controller_max_gradient_norm', type=float, default=5.0)
    parser.add_argument('--controller_time_major', action='store_true', default=True)
    parser.add_argument('--controller_symmetry', action='store_true', default=True)
    parser.add_argument('--controller_predict_beam_width', type=int, default=0)
    parser.add_argument('--controller_predict_lambda', type=float, default=1)
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--dataset', type=str, default='cifar10')
    return parser


def set_params():
    flags, unparsed = construct_parser().parse_known_args()
    Params.set_params(flags)
    return unparsed
