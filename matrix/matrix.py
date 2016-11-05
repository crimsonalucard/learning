from functools import reduce
from numbers import Number

class Matrix(object):
    def __init__(self, list_of_rows):
        if isinstance(list_of_rows, Matrix):
            self.matrix = list_of_rows.matrix
        elif isinstance(list_of_rows, list) and isinstance(list_of_rows[0], list):
            self.matrix = list_of_rows
        else:
            raise Exception("Cannot initialize matrix, type error")

    def rows(self):
        return len(self.matrix)

    def cols(self):
        return len(self.matrix[0])

    def getRow(self, n):
        return self.matrix[n]

    def getCol(self, n):
        return [[row[n]] for row in self.matrix]

    def transpose(self):
        return Matrix([[i[0] for i in self.getCol(col_index)] for col_index in range(self.cols())])

    def __add__(self, other):
        if self.cols() != other.cols() or self.rows() != other.rows():
            raise ArithmeticError
        return Matrix([[col0+col1 for col0, col1 in zip(row0,row1)] for row0, row1 in zip(self.matrix, other.matrix)])

    def __neg__(self):
        return Matrix([[-i for i in row] for row in self.matrix])

    def __sub__(self, other):
        if isinstance(other, Number):
            other = Matrix([[other for _ in row] for row in self.matrix])
        return self + (-other)

    def __rsub__(self,other):
        return -self.__sub__(other)

    def __repr__(self):
        format_numbers = lambda nums : " ".join(["{: .2f}".format(num) for num in nums])
        return "\n"+"\n".join([format_numbers(row) for row in self.matrix]) + "\n"

    def __getitem__(self, key):
        return self.matrix[key]

    def __iter__(self):
        return iter(reduce(lambda acc, row: acc + row, self.matrix, []))
    
    def __mul__(self,other):
        if self.cols() != other.cols() or self.rows() != other.rows():
            raise ArithmeticError
        return Matrix([[num1*num2 for num1, num2 in zip(row1, row2)] for row1, row2 in zip(self.matrix, other.matrix)])


    def __matmul__(self, other):
        if self.cols() != other.rows():
            raise ArithmeticError
        result = []
        for row in self.matrix:
            result_row = []
            for col_index in range(other.cols()):
                col = sum([i*j for i,j in zip(Matrix(other.getCol(col_index)).transpose().getRow(0),row)])
                result_row.append(col)
            result.append(result_row)
        return Matrix(result)
