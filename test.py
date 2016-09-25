#!/usr/bin/env python3

from neural_net.neural_net import create_neural_net, initialize_thetas, create_forward_propogator
from matrix.matrix import Matrix
from misc.utils import sigmoid

x = create_neural_net([[0,0,0],[0,0,0]], activation_function=sigmoid)
print(x(1,1))

random_stuff = initialize_thetas(2, 4, 5, 3)
print(random_stuff)

print("testing forward propogator")
y = create_forward_propogator([[0,0,0],[0,0,0]])

print(y(1,1))
