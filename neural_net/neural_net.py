#/usr/bin/env python3

from matrix.matrix import Matrix
from misc.utils import sigmoid
from random import random
from functools import reduce


#creates neural net that takes n inputs from thetas with n+1 columns
def create_neural_net(*thetas, **kwargs):
    activation_function = sigmoid if "activation_function" not in kwargs else kwargs["activation_function"] 
    thetas = [Matrix(theta) for theta in thetas]
    def neural_net(*xs):
        if len(xs) != len(thetas[0][0]) - 1:
            raise ValueError("There were {0} inputs. The neural net only accepts {1} inputs".format(len(xs), len(thetas[0][0]))) 
        xs = Matrix([xs]).transpose()
        return reduce(lambda acc, theta: activation_function(theta @ augment_column_matrix(acc,1)), thetas, xs)
    return neural_net

def augment_column_matrix(column_matrix, num):
    return Matrix(column_matrix.matrix + [[num]])



#generate random number between (-error, error)
def generate_random_number(error):
    return random()*2*error-error

def initialize_thetas(error, *layer_lengths):
    thetas = []
    for index, layer_length in enumerate(layer_lengths[:-1]):
        theta = Matrix([[generate_random_number(error) for _ in range(layer_length)] for _ in range(layer_lengths[index+1])])
        thetas.append(theta)
    return thetas

def create_forward_propogator(*thetas):
    activation_propogators = [create_neural_net(theta) for theta in thetas]
    zeta_propogators       = [create_neural_net(theta, activation_function=lambda x: x) for theta in thetas]

    #[xs] -> [(Matrix, Matrix)]
    def forward_propogator(*xs):
        xs = Matrix([list(xs)])
        result = [[activation_propogators[0](*xs), zeta_propogators[0](*xs)]]
        for activation_propogator, zeta_propogator in zip(activation_propogators[1:], zeta_propogators[1:]):
            result += [[activation_propogator(*result[-1][0]),zeta_propogator(*result[-1][0])]]
        return result
    return forward_propogator

def create_backward_propogator(*thetas, cost_function, derivative_cost, activation_function, derivative_activation):
    reversed_thetas = reversed(thetas)
    forward_propogator = create_forward_propogator(*thetas)
    
    def backward_propogate(*xs):
        propogated_values = forward_propogator(*xs)
        activations = reversed([activations for activations, _ in propogated_values])
        zetas = reversed([zeta for _, zeta in propogated_values])
        initial_error = derivative_cost(activations[0]) * derivative_activation(zetas[0])
        return reduce(lambda acc, x: [(x[0].transpose() @ acc[0]) * derivative_activation(x[1])] + acc, zip(thetas, zetas[1:]), [initial_error])


