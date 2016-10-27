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
     


def append_matrix_to_file(matrix, fileHandler):
    for row in matrix.matrix:
        for col in row:
            fileHandler.write(str(col))
            fileHandler.write(' ')
        fileHandler.write('\n')
    fileHandler.write(';\n')
    return fileHandler

def read_matrix_from_file(fileHandler):
    list_of_rows = []
    line = 'initialize' 
    line = fileHandler.readline()
    while ';' not in line:
        if line == '':
            return ''
    #    print("reading row: {0}".format(line))
        row = [int(i) for i in line.split()]
        list_of_rows.append(row)
        line = fileHandler.readline()
    #print(list_of_rows)
    return Matrix(list_of_rows)

def read_list_of_matrices(filename):
    fileHandler = open(filename, 'r')
    result = []
    matrix = "initialize!"
    while True:
        matrix = read_matrix_from_file(fileHandler) 
        if matrix == '':
            break
        result.append(matrix)
    return result

def write_list_of_matrices(list_of_matrices, fileName, extension="matrix"):
    fileHandler = open("{0}.{1}".format(fileName, extension), 'w+')
    for matrix in list_of_matrices:
        append_matrix_to_file(matrix, fileHandler)
    fileHandler.close()

@matrix_application
def sigmoid(n):
    is_negative_exponent = True if -n < 0 else False
    try:
        return 1.0/(1.0+e**(-n))
    except OverflowError: #sigmoid risks overflow if numbers are to high. Use limits. 
        if is_negative_exponent:
            return 1.0
        else:
            return 0.0

@matrix_application
def sigmoid_derivative(n):
    return sigmoid(n)*(1 - sigmoid(n))

@matrix_application
def square_error(activation_value, actual_value):
    return (0.5) * (actual_value - activation_value) ** 2 

@matrix_application
def square_error_derivative(activation_value, actual_value):
    return (actual_value - activation_function)
