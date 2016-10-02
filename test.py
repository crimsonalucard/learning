#!/usr/bin/env python3

from neural_net.neural_net import create_neural_net, initialize_thetas, create_forward_propogator, create_backward_propogator
from matrix.matrix import Matrix
from misc.utils import sigmoid, sigmoid_derivative, square_error, square_error_derivative


x = create_neural_net([[0,0,0],[0,0,0]], activation_function=sigmoid)
print(x(1,1))


print("testing forward propogator")
y = create_forward_propogator([[0,0,0],[0,0,0]])

print(y(1,1))

thetas = initialize_thetas(1, 784, 15, 10)
print(thetas)

print("testing backward propogator")
d = create_backward_propogator(thetas, square_error, square_error_derivative, sigmoid, sigmoid_derivative)
