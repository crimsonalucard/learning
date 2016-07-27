from math import e
from matrix.matrix import Matrix



#makes a function with a single parameter applicable on the Matrix class. 
def matrix_application(func):
    def func_wrapper(*parameters):
        if isinstance(parameters[0], Matrix) and len(parameters) == 1:
            return Matrix([[func(col) for col in row] for row in parameters[0]])
        else:
            return func(*parameters)

            



@matrix_application
def sigmoid(n):
    return 1/(1-e**n)
