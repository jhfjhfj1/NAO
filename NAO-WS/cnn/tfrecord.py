import os
import numpy as np
import tensorflow as tf


def ensure_dir(directory):
    """Create directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_array(data, path):
    ensure_dir(path)
    data = np.array(data)
    path = os.path.join(path, 'array')

    with tf.Graph().as_default():
        with tf.Session() as sess:
            data = tf.get_variable("data1", initializer=data)
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            # print(sess.run(data))
            saver.save(sess, path)


def load_array(path):
    meta_path = os.path.join(path, 'array.meta')
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(sess, tf.train.latest_checkpoint(path))
        # print(sess.run('data:0'))
        data = sess.run('data1:0')
        return data


if __name__ == '__main__':
    # test_output_tfrecords()
    numpy_mda = np.random.rand(2, 3, 4)
    path = 'models/test_save'
    save_array(numpy_mda, path)
    load_array(path)
    save_array(numpy_mda, path)
    load_array(path)
    save_array(numpy_mda, path)
    load_array(path)
