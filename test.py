#!/usr/bin/env python3

from neural_net.neural_net import create_neural_net, initialize_thetas, create_forward_propogator, create_backward_propogator
from matrix.matrix import Matrix
from misc.utils import sigmoid, sigmoid_derivative, square_error, square_error_derivative
from display.display import show
from mnist.reader import get_image_data, convert_flattened_array_to_matrix


x = create_neural_net([[0,0,0],[0,0,0]], activation_function=sigmoid)
print(x(1,1))


print("testing forward propogator")
y = create_forward_propogator([[0,0,0],[0,0,0]])

print(y(1,1))

thetas = initialize_thetas(1, 784, 15, 10)
print(thetas)

print("testing data retrieval")
images = get_image_data()
test_image = convert_flattened_array_to_matrix(28,28,images[0])
#import pdb; pdb.set_trace()
show(test_image, 10, "testing!")

"""
print("testing backward propogator")
d = create_backward_propogator(thetas, square_error, square_error_derivative, sigmoid, sigmoid_derivative)
"""
