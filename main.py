import numpy as np
import matplotlib.pyplot as plt
from get_data import DataInitializerMNIST
from nn import MyNeuralNetwork

data = DataInitializerMNIST()
nn = MyNeuralNetwork(data)
nn.gradient_descent()
nn.test_accuracy_with_test_data()
