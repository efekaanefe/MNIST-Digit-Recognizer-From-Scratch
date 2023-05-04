import numpy as np
import matplotlib.pyplot as plt


class MyNeuralNetwork:
    def __init__(self, dataInitializer, input=784, hidden=10, output=10):
        self.data = dataInitializer
        self.input = input
        self.hidden = hidden
        self.output = output
        self.activations = ActivationFunctions()

        self.initialize()

    def gradient_descent(
        self,
        epochs=100,
        learning_rate=0.1,
        batch_size=20000,
        print_acc=True,
        plot_acc=True,
    ):
        train_X_flatten = self.data.train_X_flatten
        train_y_onehot = self.data.train_y_onehot

        iterations = train_X_flatten.T.shape[0] // batch_size

        self.accuracy_values = []
        self.epoch_values = []

        for epoch in range(epochs):
            for iteration in range(iXterations):

                # parsing the data
                index0 = iteration * batch_size
                index1 = (iteration + 1) * batch_size

                X = train_X_flatten.T[index0:index1].T
                Y = train_y_onehot.T[index0:index1].T

                Z1, A1, Z2, A2 = self.forward_propagation(X)
                dW1, db1, dW2, db2 = self.backward_propagation(X, Y, Z1, A1, Z2, A2)
                self.update_parameters(learning_rate, dW1, db1, dW2, db2)

                if print_acc:
                    prediction, accuracy = print_accuracy(
                        A2, self.data.train_y.T[index0:index1]
                    )
                    self.accuracy_values.append(accuracy)
                    self.epoch_values.append(epoch)

        if plot_acc:
            title = f"accuracy vs epoch = {epoch}, batch_size = {batch_size}, learning_rate = {learning_rate}, iterations = {iterations}"
            plot_accuracy(self.accuracy_values, self.epoch_values, title)

    def print_accuracy(self, A2, y):
        print("Epoch:", epoch + 1)
        predictions = get_predictions(A2)
        accuracy = get_accuracy(
            predictions,
            train_y.T[(iteration * batch_size) : (iteration + 1) * batch_size].T,
        )
        print(accuracy)
        return predictions, accuracy

    def plot_accuracy(self, accuracy_values, epoch_values, title):
        fig = plt.figure(1)  # identifies the figure
        plt.title(title, fontsize="16")  # title
        plt.plot(epoch_values, accuracy_values)  # plot the points
        plt.xlabel("epoch", fontsize="13")  # adds a label in the x axis
        plt.ylabel("accuracy", fontsize="13")  # adds a label in the y axis
        # plt.savefig(f"epoch_{epoch} batch_size_{batch_size}.png")	#saves the figure in the present directory

        plt.grid()  # shows a grid under the plot
        plt.show()

    def initialize():
        self.W1 = np.random.uniform(-0.5, 0.5, (self.hidden, self.input))
        self.b1 = np.random.uniform(-0.5, 0.5, (self.hidden, 1))
        self.W2 = np.random.uniform(-0.5, 0.5, (self.output, self.hidden))
        self.b2 = np.random.uniform(-0.5, 0.5, (self.output, 1))

    def forward_propagation(self, X):
        Z1 = self.W1 @ X + self.b1
        A1 = self.activations.ReLU(Z1)
        Z2 = self.W2 @ A1 + self.b2
        A2 = self.activations.softmax(Z2)
        return Z1, A1, Z2, A2

    def backward_propagation(self, X, Y, Z1, A1, Z2, A2):
        dZ2 = A2 - Y
        dW2 = 1 / m * dZ2 @ A1.T
        db2 = 1 / m * np.sum(dZ2)
        dZ1 = W2.T @ dZ2 * self.activations.ReLU_deriv(Z1)
        dW1 = 1 / m * dZ1 @ X.T
        db1 = 1 / m * np.sum(dZ1)
        return dW1, db1, dW2, db2

    def update_parameters(self, learning_rate, dW1, db1, dW2, db2):
        self.W1 = self.W1 - learning_rate * dW1
        self.b1 = self.b1 - learning_rate * db1
        self.W2 = self.W2 - learning_rate * dW2
        self.b2 = self.b2 - learning_rate * db2

    def get_predictions(self, A2):
        return np.argmax(A2, 0)

    def get_accuracy(self, predictions, Y, print_predictions=False):
        if print_predictions:
            print(predictions, Y)
        return np.sum(predictions == Y) / Y.size

    def plot_and_label_X(i):
        print("Label:", self.data.train_y[i])
        print("Y onehot:", self.data.train_y_onehot.T[i])
        plt.gray()
        plt.matshow(self.data.train_X[i])
        plt.show()

    def test_accuracy_with_test_data(self):
        X = self.data.test_X_flatten
        y = self.data.test_y

        Z1, A1, Z2, A2 = self.forward_propagation(X)

        accuracy = self.get_accuracy(get_predictions(A2), y)
        print(f"Test data accuracy: {accuracy}")

    def test_with_random_data(self):
        index = np.random.randint(0, 1000)

        X = self.data.train_X_flatten.T[index : index + 1].T
        # y = self.data.train_y_onehot.T[index : index + 1].T

        Z1, A1, Z2, A2 = self.forward_propagation(X)
        print(
            f"I am % {np.around(np.max(A2)*100, 2)} certain that it is: ", np.argmax(A2)
        )
        plot_and_label_train_X(index)


class ActivationFunctions:
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(x):
        return sigmoid(x) * (1 - sigmoid(x))

    def ReLU(Z):
        return np.maximum(Z, 0)

    def ReLU_deriv(Z):
        return Z > 0

    def softmax(Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A
