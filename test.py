#!/usr/bin/env python3

from neural_net.neural_net import create_neural_net, initialize_thetas, create_forward_propogator, create_backward_propogator
from matrix.matrix import Matrix
from misc.utils import sigmoid, sigmoid_derivative, square_error, square_error_derivative, write_list_of_matrices, read_list_of_matrices, flatten_matrix, create_data_scaler
from mnist.reader import get_image_data, get_label_data, convert_flattened_array_to_matrix


#write_list_of_matrices([Matrix([[1,1],[2,2]]), Matrix([[3,3],[4,4]])], "test")
#x = read_list_of_matrices("test.matrix")
#print(x)

#x = create_neural_net([[0,0,0],[0,0,0]], activation_function=sigmoid)
#print(x(1,1))


#print("testing forward propogator")
#y = create_forward_propogator([[0,0,0],[0,0,0]])

thetas = initialize_thetas(1, 784, 30, 10)
print(thetas)
data_scaler = create_data_scaler(255)
print("testing data retrieval")
input_matrix = data_scaler(read_list_of_matrices("first_image.matrix")[0])
print("input matrix: ")
print(input_matrix)
linear_input_matrix = flatten_matrix(input_matrix)
print("flattened input matrix: ")
print(linear_input_matrix)


forward_propogator = create_forward_propogator(*thetas)
print("printing forward propagation results")
print(forward_propogator(*linear_input_matrix))

labels = get_label_data()
print(labels[0])
