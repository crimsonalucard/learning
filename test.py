#!/usr/bin/env python

from neural_net.neural_net import create_neural_net, create_forward_propogater, initialize_thetas

x = create_neural_net([[0,0,0],[0,0,0]],[[0,0,0],[0,0,0]])
print(x(1,1))

random_stuff = initialize_thetas(2, 4, 5, 3)
print(random_stuff)
