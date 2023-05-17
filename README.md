# MNIST DIGIT RECOGNIZER FROM SCRATCH

A digit recognizer neural network trained on the MNIST dataset. The neural network is implemented from scratch using only numpy.

Example:

https://github.com/efekaanefe/MNIST-Digit-Recognizer-From-Scratch/assets/81916806/5dfca99d-18f1-40dd-83b1-ff3c099e3e93

The neural network class allows different sizes of layers (input, hidden, output), however it is restricted to have only one hidden layer.
Gradient descent function accepts variable such as epochs, learning_rate,  batch_size. 
Also, by adjusting the flags (print_acc and plot_acc) you can see the training accuracy. 

## Possible Improvements
- [ ] Loss while training can be calculated.
- [ ] Multiple hidden layers can be added. For example, MyNeuralNetwork class takes hidden = (20,20,10) as input and adjusts weights and biases according to.
- [ ] Neural network can be converted to convolutional neural network.
- [ ] Another GUI can be added which gets a random value in the data and shows what the neural network guess is and the actual label.
- [ ] Other datasets such as ([EMNIST](https://github.com/hosford42/EMNIST), [fashion-mnist](https://github.com/zalandoresearch/fashion-mnist), [
quickdraw-dataset](https://github.com/googlecreativelab/quickdraw-dataset) ) can be trained.


Inspired by:
[Sebastian Lague](https://youtu.be/w8yWXqWQYmU), [Samson Zhang](https://youtu.be/hfMk-kjRv4c)

