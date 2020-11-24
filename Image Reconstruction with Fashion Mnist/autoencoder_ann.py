import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').disabled = True

import pandas as pd
import tensorflow as tf 

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("\n Num GPUs Available: {}\n".format(len(physical_devices)))
print(" Tensorflow Version: {}\n".format(tf.__version__))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from variables import*
from util import*
tf.random.set_seed(seed);

class ImageReconstruction(object):
    def __init__(self):
        TFtrain, TFval, TFtest = get_data()
        self.TFtrain = TFtrain
        self.TFtest = TFtest
        self.TFval = TFval

        self.optimizer = tf.keras.optimizers.Adam()

        self.train_loss_obj = tf.keras.losses.MeanSquaredError()
        self.val_loss_obj = tf.keras.losses.MeanSquaredError()
        self.test_loss_obj = tf.keras.losses.MeanSquaredError()

        self.device = '/gpu:0' if tf.test.is_gpu_available() else '/cpu:0'
        print("device : {}".format(self.device))

    def build(self):
        self.model = tf.keras.models.Sequential([
                                    tf.keras.layers.Dense(dense1, input_shape=input_shape, activation='relu'),
                                    tf.keras.layers.Dense(dense2, activation='relu'),
                                    tf.keras.layers.Dense(dense2, activation='relu'),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(dense3, activation='relu'),
                                    tf.keras.layers.Dense(dense3, activation='relu'),
                                    tf.keras.layers.Dense(dense3, activation='relu'),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(dense2, activation='relu'),
                                    tf.keras.layers.Dense(dense2, activation='relu'),
                                    tf.keras.layers.Dense(dense1, activation='relu'),
                                    tf.keras.layers.Dropout(keep_prob),
                                    tf.keras.layers.Dense(input_shape[0], activation='relu'),
                                    ])
        self.model.summary()

    def train_epoch(self, X):
        with tf.GradientTape() as tape:
            P = self.model(X)
            loss = self.train_loss_obj(X, P)

        gradients = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

        return loss

    def train(self):
        for epoch in range(1, epoches+1):
            Train_loss = []
            Val_loss = []
            for Xs, _ in self.TFtrain:
                with tf.device(device_name=self.device):
                    train_loss = self.train_epoch(Xs)
                    Train_loss.append(train_loss)
                
            with tf.device(device_name=self.device):
                for Xs, _ in self.TFval:
                    P = self.model(Xs)
                    val_loss = self.val_loss_obj(Xs, P)
                    Val_loss.append(val_loss)

            avg_train_loss = round(float(tf.reduce_mean(Train_loss)), 3)
            avg_val_loss = round(float(tf.reduce_mean(Val_loss)), 3)
            print("Epoch : {}, train loss : {}, val loss : {}".format(
                                                                epoch, avg_train_loss, avg_val_loss
                                                                ))

    def save_and_load(self):
        if os.path.exists(model_path):
            print("model loading")
            loaded = tf.saved_model.load(model_path)
            self.model = loaded.signatures["serving_default"]
        else:
            print("model saving")
            self.build()
            self.train()
            tf.saved_model.save(self.model, model_path)

    def evaluation(self):
        Test_loss = []
        with tf.device(device_name=self.device):
            for Xs, _ in self.TFtest:
                P = self.model(Xs)
                test_loss = self.test_loss_obj(Xs, P)
                Test_loss.append(test_loss)

        avg_test_loss = round(float(tf.reduce_mean(Test_loss)), 3)
        print("\n----Evaluation----")
        print("test loss : {}".format(avg_test_loss))

    def predictions(self):
        img = get_random_item(self.TFtest)
        Pimg= self.model(img)
        
        img = img.numpy()
        Pimg = Pimg.numpy()

        img = img.reshape(size, size)
        Pimg = Pimg.reshape(size, size)
        plot_item(Ximg, Yimg)
    
    def run(self):
        self.save_and_load()
        self.evaluation()
        self.predictions()

if __name__ == "__main__":
    model = ImageReconstruction()
    model.run()