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

class FashionMnist(object):
    def __init__(self):
        TFtrain, TFval, TFtest = get_data()
        self.TFtrain = TFtrain
        self.TFtest = TFtest
        self.TFval = TFval

        self.optimizer = tf.keras.optimizers.Adam()

        self.train_loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
        self.train_accuracy_obj = tf.keras.metrics.SparseCategoricalAccuracy()

        self.val_loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
        self.val_accuracy_obj = tf.keras.metrics.SparseCategoricalAccuracy()

        self.test_loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
        self.test_accuracy_obj = tf.keras.metrics.SparseCategoricalAccuracy()

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
                                    tf.keras.layers.Dropout(keep_prob),
                                    tf.keras.layers.Dense(output, activation='softmax'),
                                    ])
        self.model.summary()

    def train_epoch(self, X, Y):
        with tf.GradientTape() as tape:
            P = self.model(X)
            loss = self.train_loss_obj(Y, P)

        gradients = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

        self.train_accuracy_obj.update_state(Y, P)

        return loss

    def train(self):
        for epoch in range(1, epoches+1):
            Train_loss = []
            Val_loss = []
            for Xs, Ys in self.TFtrain:
                with tf.device(device_name=self.device):
                    train_loss = self.train_epoch(Xs, Ys)
                    Train_loss.append(train_loss)

            train_acc = self.train_accuracy_obj.result().numpy()
                
            with tf.device(device_name=self.device):
                for Xs, Ys in self.TFval:
                    P = self.model(Xs)
                    val_loss = self.val_loss_obj(Ys, P)
                    Val_loss.append(val_loss)
                    self.val_accuracy_obj.update_state(Ys, P)

            val_acc = self.val_accuracy_obj.result().numpy()

            avg_train_loss = round(float(tf.reduce_mean(Train_loss)), 3)
            avg_val_loss = round(float(tf.reduce_mean(Val_loss)), 3)
            train_acc = round(float(train_acc), 3)
            val_acc = round(float(val_acc), 3)
            print("Epoch : {}, train loss : {}, train accuracy : {}, val loss : {}, val accuracy : {}".format(
                                                                epoch, avg_train_loss, train_acc, avg_val_loss, val_acc
                                                                ))

            self.train_accuracy_obj.reset_states()
            self.val_accuracy_obj.reset_states()

    def evaluation(self):
        Test_loss = []
        with tf.device(device_name=self.device):
            for Xs, Ys in self.TFtest:
                P = self.model(Xs)
                test_loss = self.test_loss_obj(Ys, P)
                Test_loss.append(test_loss)
                self.test_accuracy_obj.update_state(Ys, P)

        test_acc = self.test_accuracy_obj.result().numpy()  
        test_acc = round(float(test_acc), 3)  
        avg_test_loss = round(float(tf.reduce_mean(Test_loss)), 3)
        self.test_accuracy_obj.reset_states()
        print("\n---------------Evaluation----------------")
        print("test loss : {}, test acc : {}".format(avg_test_loss, test_acc))

if __name__ == "__main__":
    model = FashionMnist()
    model.build()
    model.train()
    model.evaluation()