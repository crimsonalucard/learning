from matrix.matrix import Matrix
from misc.utils import sigmoid

def create_neural_net(*thetas):
    thetas = [Matrix(theta) for theta in thetas]
    def neural_net(*xs):
        xs = Matrix([xs]).transpose()
        return reduce(lambda acc, theta: sigmoid(theta*acc), thetas, xs)
    return neural_net
        


