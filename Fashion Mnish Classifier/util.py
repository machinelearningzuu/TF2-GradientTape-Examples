import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').disabled = True

import pandas as pd
import tensorflow as tf 

from variables import*
tf.random.set_seed(seed);

def load_dataframe(csv_path):
    df = pd.read_csv(csv_path)
    labels = df.pop('label').values
    images = df.values
    images = images / 255.0
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