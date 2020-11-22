import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').disabled = True

import tensorflow as tf 

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("\n Num GPUs Available: {}\n".format(len(physical_devices)))
print(" Tensorflow Version: {}\n".format(tf.__version__))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from variables import*
from util import*
tf.random.set_seed(seed);

class PolynomialRegression(object):
    def __init__(self):
        train_dataset, test_dataset = load_poly_data()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.optimizer = tf.keras.optimizers.Adam()
        self.train_loss_obj = tf.keras.losses.MeanSquaredError()
        self.test_loss_obj = tf.keras.losses.MeanSquaredError()
        self.device = '/gpu:0' if tf.test.is_gpu_available() else '/cpu:0'
        print("device : {}".format(self.device))

    def build(self):
        model = tf.keras.Sequential([
                            tf.keras.layers.Flatten(input_shape=input_shape),
                            tf.keras.layers.Dense(dense1, activation='relu'),
                            tf.keras.layers.Dense(dense2, activation='relu'),
                            tf.keras.layers.Dense(1)
                            ], name='Polynomial_Regression_Model')
        self.model = model 
        self.model.summary()

    def train_epoch(self, X, Y):
        with tf.GradientTape() as tape:
            P = self.model(X)
            loss = self.train_loss_obj(Y, P)

        gradients = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        return loss

    def train(self):
        for epoch in range(1, epoches+1):
            Train_loss = []
            Test_Loss = []
            for Xs, Ys in self.train_dataset:
                with tf.device(device_name=self.device):
                    train_loss = self.train_epoch(Xs, Ys)
                    Train_loss.append(train_loss)
                
            with tf.device(device_name=self.device):
                for Xs, Ys in self.test_dataset:
                    P = self.model(x)
                    test_loss = self.test_loss_obj(Ys, P)
                    Test_Loss.append(test_loss)

            avg_train_loss = round(float(tf.reduce_mean(Train_loss)), 3)
            avg_test_loss = round(float(tf.reduce_mean(Train_loss)), 3)
            print("Epoch : {}, train loss : {}, test loss : {}".format(epoch, avg_train_loss, avg_test_loss))

if __name__ == "__main__":
    model = PolynomialRegression()
    model.build()
    model.train()
