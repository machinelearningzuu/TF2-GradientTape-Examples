import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').disabled = True

import tensorflow as tf 
from variables import*

tf.random.set_seed(seed);

def linear_func(x):
    return 2*x + 1

def load_linear_data():
    X = tf.random.normal(
                    [n_samples,],
                    seed=seed,
                    dtype=tf.float32
                    )
    Y = linear_func(X)

    TFdataset = tf.data.Dataset.from_tensor_slices((X, Y))
    TFdataset = TFdataset.shuffle(shuffle_buffer_size).batch(batch_size)
    train_dataset = TFdataset.take(n_train)
    test_dataset = TFdataset.skip(n_train).take(n_samples-n_train)
    return train_dataset, test_dataset
