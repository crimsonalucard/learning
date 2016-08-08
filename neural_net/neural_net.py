from matrix.matrix import Matrix
from misc.utils import sigmoid
from random import random

def create_neural_net(*thetas):
    thetas = [Matrix(theta) for theta in thetas]
    def neural_net(*xs):
        xs = Matrix([xs]).transpose()
        return reduce(lambda acc, theta: sigmoid(theta*acc), thetas, xs)
    return neural_net

#generate random number between (-error, error)
def generate_random_number(error):
    return random()*2*error-error

def initialize_thetas(error, *layer_lengths):
    thetas = []
    for index, layer_length in enumerate(layer_lengths[:-1]):
        theta = Matrix([[generate_random_number(error) for node in layer_length] for next_layer in layer_lengths[index+1]])
        thetas.append(theta)
    return thetas

def create_forward_propogater(thetas):
    neural_nets = [create_neural_net(theta) for theta in thetas]
    def forward_propagater(*xs):
        xs = Matrix([xs]).transpose()
        return reduce(lambda acc, neural_net: acc+[neural_net(acc[-1])], neural_nets, xs)
    return forward_propagater

