import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').disabled = True

import numpy as np
import pandas as pd
import tensorflow as tf 
import matplotlib.pyplot as plt

from variables import*
tf.random.set_seed(seed);

def load_dataframe(csv_path):
    df = pd.read_csv(csv_path)
    labels = df.pop('label').values
    images = df.values
    images = images / 255.0
    images = images.astype('float32')
    return images, labels 

def get_data():
    Xtrain, Ytrain = load_dataframe(train_path)
    Xtest, Ytest = load_dataframe(test_path)
    Ntrain = int(len(Ytrain) * split) // batch_size

    trainDS = tf.data.Dataset.from_tensor_slices((Xtrain, Ytrain))
    testDS = tf.data.Dataset.from_tensor_slices((Xtest, Ytest))

    trainDS = trainDS.shuffle(shuffle_buffer_size).batch(batch_size)

    TFtest = testDS.shuffle(shuffle_buffer_size).batch(batch_size)
    TFtrain = trainDS.take(Ntrain)
    TFval = trainDS.skip(Ntrain)
    return TFtrain, TFval, TFtest

def get_random_item(TFtest):
    n_batches = len(list(TFtest))

    rand_batch_idx = np.random.choice(n_batches)
    rand_batch = list(TFtest)[rand_batch_idx]

    batch_image = rand_batch[0]
    batch_label = rand_batch[1]

    image_idx = np.random.choice(len(batch_image))
    image = batch_image[image_idx]
    label = batch_label[image_idx]
    image = tf.expand_dims(
                        image, 
                        axis=0
                        )
    image = tf.cast(image, dtype=tf.float32)
    return image, label

def plot_item(Ximg, Yimg, plt_img, width=5, height=5, rows = 1, cols = 2):
    f = plt.figure()

    f.add_subplot(1,2, 1)
    plt.title('Original Image')
    plt.imshow(Ximg)

    f.add_subplot(1,2, 2)
    plt.title('Reconstructed Image')
    plt.imshow(Yimg)
    plt.savefig(plt_img)
    plt.show(block=True)