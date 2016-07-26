class Matrix(object):
    def __init__(self, list_of_rows):
        self.matrix = list_of_rows

    def rows(self):
        return len(self.matrix)

    def cols(self):
        return len(self.matrix[0])

    def getRow(self, n):
        return self.matrix[n]

    def getCol(self, n):
        return [[row[n]] for row in matrix]

    def transpose(self):
        return Matrix([[i[0] for i in self.getCol(col_index)] for col_index in range(cols)])

    def __add__(self, other):
        if self.cols() != other.cols() or self.row() != other.row():
            raise ArithmeticError
        return Matrix([[col0+col1 for col0, col1 in zip(row0,row1)] for row0, row1 in zip(self.matrix, other.matrix)])

    def __neg__(self):
        return Matrix([[-i for i in self.row] for row in self.matrx])

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if self.rows() != other.cols():
            raise ArithmeticError
        result = []
        for rows in self.matrix:
            row = []
            for col_index in range(other.cols()):
                col = sum([i*j for i,j in zip(Matrix(other.getCol(col_index)).transpose().matrix ,row)])
                row.append(col)
            result.append(row)
        return result

        

