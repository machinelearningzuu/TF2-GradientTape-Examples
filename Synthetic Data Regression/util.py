import os
import logging
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').disabled = True

import tensorflow as tf 
from variables import*

tf.random.set_seed(seed);

def linear_func(x):
    return 2*x + 1

def poly_func(x):
    return x**2 + 1

def plot_data(X, Y):
    X_np = X.numpy()
    Y_np = Y.numpy()
    plt.scatter(X_np, Y_np)
    plt.show()

def load_linear_data():
    X = tf.random.normal(
                    [n_samples,],
                    seed=seed,
                    dtype=tf.float32
                    )
    Y = linear_func(X)
    Y_rand = tf.random.uniform(
                            [n_samples,],
                            minval=-1, 
                            maxval=1,
                            seed=seed,
                            dtype=tf.float32)
    Y = tf.add(Y, Y_rand)
    plot_data(X, Y)
    TFdataset = tf.data.Dataset.from_tensor_slices((X, Y))
    TFdataset = TFdataset.shuffle(shuffle_buffer_size).batch(batch_size)
    train_dataset = TFdataset.take(n_train)
    test_dataset = TFdataset.skip(n_train).take(n_samples-n_train)
    return train_dataset, test_dataset

def load_poly_data():
    X = tf.random.normal(
                    [n_samples,],
                    seed=seed,
                    dtype=tf.float32
                    )
    Y = poly_func(X)
    Y_rand = tf.random.uniform(
                            [n_samples,],
                            minval=-1, 
                            maxval=1,
                            seed=seed,
                            dtype=tf.float32)
    Y = tf.add(Y, Y_rand)
    plot_data(X, Y)
    TFdataset = tf.data.Dataset.from_tensor_slices((X, Y))
    TFdataset = TFdataset.shuffle(shuffle_buffer_size).batch(batch_size)
    train_dataset = TFdataset.take(n_train)
    test_dataset = TFdataset.skip(n_train).take(n_samples-n_train)
    return train_dataset, test_dataset
