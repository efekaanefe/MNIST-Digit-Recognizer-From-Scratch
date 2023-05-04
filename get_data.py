from keras.datasets import mnist


class DataInitializerMNIST:
    def __init__(self):
        self.load_mnist_data()
        self.normalize_X()
        self.get_X_flatten_data()
        self.get_y_one_hot_data()

    def load_mnist_data(self):
        (self.train_X, self.train_y), (self.test_X, self.test_y) = mnist.load_data()

    def normalize_X(self):
        self.train_X / 255
        self.test_X / 255

    def get_y_one_hot_data(self):
        self.train_y_onehot = get_one_hot_y(self.train_y)  # shape -> (10,60000)
        self.test_y_onehot = get_one_hot_y(self.test_y)  # shape -> (10,60000)

    def get_X_flatten_data(self):
        self.train_X_flatten = get_flatten_X(self.train_X)  # shape -> (784,60000)
        self.test_X_flatten = get_flatten_X(self.test_X)  # shape -> (784,60000)

    def get_one_hot_y(self, y):
        output = []
        for i in range(y.shape[0]):
            tmp = np.array([0] * 10)
            tmp[train_y[i]] = 1
            output.append(tmp)
        return np.array(output).T

    def get_flatten_X(self, X):
        output = []
        for i in range(X.shape[0]):
            output.append(X[i].flatten())
        return np.array(output).T
