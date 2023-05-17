from keras.datasets import mnist
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


class DataInitializerMNIST:
    def __init__(self):
        self.load_mnist_data()
        self.add_augmented_data()
        self.normalize_X()
        self.get_X_flatten_data()
        self.get_y_one_hot_data()

    def load_mnist_data(self):
        (self.train_X, self.train_y), (self.test_X, self.test_y) = mnist.load_data()

    def normalize_X(self):
        self.train_X = self.train_X / 255
        self.test_X = self.test_X / 255

    def get_y_one_hot_data(self):
        self.train_y_onehot = self.get_one_hot_y(self.train_y)  # shape -> (10,60000)
        self.test_y_onehot = self.get_one_hot_y(self.test_y)  # shape -> (10,60000)

    def get_X_flatten_data(self):
        self.train_X_flatten = self.get_flatten_X(self.train_X)  # shape -> (784,60000)
        self.test_X_flatten = self.get_flatten_X(self.test_X)  # shape -> (784,60000)

    def get_one_hot_y(self, y):
        output = []
        for i in range(y.shape[0]):
            tmp = np.array([0] * 10)
            tmp[self.train_y[i]] = 1
            output.append(tmp)
        return np.array(output).T

    def get_flatten_X(self, X):
        output = []
        for i in range(X.shape[0]):
            output.append(X[i].flatten())
        return np.array(output).T

    def add_augmented_data(self):
        datagen = ImageDataGenerator(
            rotation_range=10,  # randomly rotate images by X degrees
            width_shift_range=0.1,  # randomly shift images horizontally by X%
            height_shift_range=0.1,  # randomly shift images vertically by X%
            zoom_range=0.1,  # randomly zoom images by up to X%
            fill_mode="nearest",  # fill in missing pixels with nearest value
        )

        X_train = self.train_X.reshape(self.train_X.shape[0], 28, 28, 1)
        datagen.fit(X_train)

        aug_X_train = datagen.flow(X_train, batch_size=60000, shuffle=False).next()
        aug_X_train = aug_X_train.reshape(X_train.shape[0], 28, 28)

        # self.train_X = np.append(self.train_X, aug_X_train, axis=0)
        # self.train_y = np.append(self.train_y, self.train_y, axis=0)

        self.train_X = aug_X_train
