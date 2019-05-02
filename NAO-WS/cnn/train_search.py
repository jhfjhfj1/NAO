from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
import math

from .controller import predict, train
from .params import set_params, Params
from .last import augment
from .model_search import train as child_train
from .model_search import valid as child_valid
from .calculate_params import calculate_params
from .utils import generate_arch, parse_arch_to_seq, parse_seq_to_arch


def _log_variable_sizes(var_list, tag):
    """Log the sizes and shapes of variables, and the total size.

    Args:
      var_list: a list of varaibles
      tag: a string
  """
    name_to_var = {v.name: v for v in var_list}
    total_size = 0
    for v_name in sorted(list(name_to_var)):
        v = name_to_var[v_name]
        v_size = int(np.prod(np.array(v.shape.as_list())))
        tf.logging.info("Weight    %s\tshape    %s\tsize    %d",
                        v.name[:-2].ljust(80),
                        str(v.shape).ljust(20), v_size)
        total_size += v_size
    tf.logging.info("%s Total size: %d", tag, total_size)


def search_train():
    # child_params = get_child_model_params()
    # controller_params = get_controller_params()
    branch_length = Params.source_length // 2 // 5 // 2
    eval_every_epochs = Params.eval_every_epochs
    child_epoch = 0
    while True:
        # Train child model
        if Params.arch_pool is None:
            arch_pool = generate_arch(Params.num_seed_arch, Params.num_cells,
                                            5)  # [[[conv],[reduc]]]
            Params.arch_pool = arch_pool
            Params.arch_pool_prob = None
        else:
            if Params.sample_policy == 'uniform':
                Params.arch_pool_prob = None
            elif Params.sample_policy == 'params':
                Params.arch_pool_prob = calculate_params(Params.arch_pool)
            elif Params.sample_policy == 'valid_performance':
                Params.arch_pool_prob = child_valid()
            elif Params.sample_policy == 'predicted_performance':
                encoder_input = list(map(lambda x: parse_arch_to_seq(x[0], branch_length) + \
                                                   parse_arch_to_seq(x[1], branch_length),
                                         Params.arch_pool))
                predicted_error_rate = predict(encoder_input)
                Params.arch_pool_prob = [1 - i[0] for i in predicted_error_rate]
            else:
                raise ValueError('Child model arch pool sample policy is not provided!')

        if isinstance(eval_every_epochs, int):
            Params.eval_every_epochs = eval_every_epochs
        else:
            for index, e in enumerate(eval_every_epochs):
                if child_epoch < e:
                    Params.eval_every_epochs = e
                    break

        child_epoch = child_train()

        # Evaluate seed archs
        valid_accuracy_list = child_valid()

        # Output archs and evaluated error rate
        old_archs = Params.arch_pool
        old_archs_perf = [(1 - acc) for acc in valid_accuracy_list]

        # Old archs are sorted.
        old_archs_sorted_indices = np.argsort(old_archs_perf)
        old_archs = np.array(old_archs)[old_archs_sorted_indices].tolist()
        old_archs_perf = np.array(old_archs_perf)[old_archs_sorted_indices].tolist()
        child_model_dir = Params.get_child_model_dir()
        with open(os.path.join(child_model_dir, 'arch_pool.{}'.format(child_epoch)), 'w') as fa:
            with open(os.path.join(child_model_dir, 'arch_pool.perf.{}'.format(child_epoch)), 'w') as fp:
                with open(os.path.join(child_model_dir, 'arch_pool'), 'w') as fa_latest:
                    with open(os.path.join(child_model_dir, 'arch_pool.perf'), 'w') as fp_latest:
                        for arch, perf in zip(old_archs, old_archs_perf):
                            arch = ' '.join(map(str, arch[0] + arch[1]))
                            fa.write('{}\n'.format(arch))
                            fa_latest.write('{}\n'.format(arch))
                            fp.write('{}\n'.format(perf))
                            fp_latest.write('{}\n'.format(perf))

        print(child_epoch)
        import pickle
        pickle.dump((old_archs, old_archs_perf), open(os.path.join(child_model_dir,
                                                                   'history.{}'.format(child_epoch)),
                                                      'wb'))
        if child_epoch >= Params.num_epochs:
            break

        # Train Encoder-Predictor-Decoder
        encoder_input = list(map(lambda x: parse_arch_to_seq(x[0], branch_length) + \
                                           parse_arch_to_seq(x[1], branch_length), old_archs))
        # [[conv, reduc]]
        # Normalization: Normalize the architecture performances to 0 to 1.
        min_val = min(old_archs_perf)
        max_val = max(old_archs_perf)
        predictor_target = [(i - min_val) / (max_val - min_val) for i in old_archs_perf]
        Params.batches_per_epoch = math.ceil(len(encoder_input) / Params.controller_batch_size)
        # if clean controller model
        if Params.augment:
            history = augment(encoder_input, predictor_target)
            for archs, perfs, _ in history:
                encoder_input += archs
                predictor_target += perfs
        train(encoder_input, predictor_target)

        # Generate new archs
        # old_archs = old_archs[:450]
        new_archs = []
        max_step_size = 100
        Params.predict_lambda = 0
        top100_archs = list(map(lambda x: parse_arch_to_seq(x[0], branch_length) + \
                                          parse_arch_to_seq(x[1], branch_length), old_archs[:100]))

        new_arch_lower_bound = int(Params.num_seed_arch / 2)
        while len(new_archs) < new_arch_lower_bound:
            Params.predict_lambda += 1
            new_arch = predict(top100_archs)
            for arch in new_arch:
                if arch not in encoder_input and arch not in new_archs:
                    new_archs.append(arch)
                if len(new_archs) >= new_arch_lower_bound:
                    break
            tf.logging.info('{} new archs generated now'.format(len(new_archs)))
            if Params.predict_lambda > max_step_size:
                break
                # [[conv, reduc]]
        new_archs = list(map(lambda x: parse_seq_to_arch(x, branch_length), new_archs))  # [[[conv],[reduc]]]
        num_new_archs = len(new_archs)
        tf.logging.info("Generate {} new archs".format(num_new_archs))
        # The pool size is always the same as the original seeds : 1000. Every time just replace the last ones.
        new_arch_pool = old_archs[:len(old_archs) - (num_new_archs + int(new_arch_lower_bound / 10))] + new_archs + generate_arch(int(new_arch_lower_bound / 10), 5, 5)
        tf.logging.info("Totally {} archs now to train".format(len(new_arch_pool)))
        Params.arch_pool = new_arch_pool
        with open(os.path.join(child_model_dir, 'arch_pool'), 'w') as f:
            for arch in new_arch_pool:
                arch = ' '.join(map(str, arch[0] + arch[1]))
                f.write('{}\n'.format(arch))


def main(unused_argv):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    # all_params = vars(Params)
    # with open(os.path.join(Params.output_dir, 'hparams.json'), 'w') as f:
    #     json.dump(all_params, f)
    search_train()


if __name__ == '__main__':
    import os

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
    tf.logging.set_verbosity(tf.logging.INFO)
    unparsed = set_params()
    tf.app.run(argv=[sys.argv[0]] + unparsed)
