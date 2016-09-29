from math import e
from matrix.matrix import Matrix
from functools import reduce

def concat_matrices_horizontal(*matrices):
    return Matrix(reduce(lambda acc, matrix: [row+acc_row for acc_row, row in zip(acc, matrix.matrix)], matrices[1:], matrices[0].matrix))

def concat_matrices_vertical(*matrices):
    pass


#makes a num -> num function into Matrix -> Matrix
def matrix_application(func):
    def func_wrapper(*parameters):
        if reduce(lambda acc, x: x and acc, [isinstance(parameter, Matrix) for parameter in parameters], True):
            return Matrix([[func(*cols) for cols in zip(*rows)] for rows in zip(*[i.matrix for i in parameters])])
        else:
            return func(*parameters)
    return func_wrapper
     



@matrix_application
def sigmoid(n):
    return 1/(1+e**n)

@matrix_application
def sigmoid_derivative(n):
    return sigmoid(n)*(1 - sigmoid(n))

@matrix_application
def square_error(activation_value, actual_value):
    return (0.5) * (actual_value - activation_value) ** 2 

@matrix_application
def square_error_derivative(activation_value, actual_value):
    return (actual_value - activation_function)
