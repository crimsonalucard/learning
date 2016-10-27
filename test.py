#!/usr/bin/env python3

from neural_net.neural_net import create_neural_net, initialize_thetas, create_forward_propogator, create_backward_propogator
from matrix.matrix import Matrix
from misc.utils import sigmoid, sigmoid_derivative, square_error, square_error_derivative, write_list_of_matrices, read_list_of_matrices
from mnist.reader import get_image_data, convert_flattened_array_to_matrix

"""
x = create_neural_net([[0,0,0],[0,0,0]], activation_function=sigmoid)
print(x(1,1))


print("testing forward propogator")
y = create_forward_propogator([[0,0,0],[0,0,0]])

print(y(1,1))

#add one for bias node
thetas = initialize_thetas(1, 785, 16, 11)
print(thetas)

print("testing data retrieval")

images = get_image_data()
forward_propogator = create_forward_propogator(*thetas)
print(forward_propogator(*images[0]))
"""

write_list_of_matrices([Matrix([[1,1],[2,2]]), Matrix([[3,3],[4,4]])], "test")
x = read_list_of_matrices("test.matrix")
print(x)


