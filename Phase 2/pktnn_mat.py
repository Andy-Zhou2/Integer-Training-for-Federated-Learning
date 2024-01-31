import numpy as np


class PktMat:
    def init_zeros(self, row, col):
        self.row = row
        self.col = col
        self.mat = np.zeros((row, col), dtype=np.int64)

    def __init__(self, row=0, col=0):
        self.init_zeros(row, col)

        self.prev_layer = None
        self.next_layer = None
        self.dummy3d = None

    def __getitem__(self, key):
        # does not need __setitem__ because we can use self.mat[key] = value
        # it returns a numpy reference
        return self.mat[key]

    def __setitem__(self, key, value):
        # it returns a numpy reference
        self.mat[key] = value

    def get_max_index_in_row(self, row):
        return self.mat[row].argmax()

if __name__ == '__main__':
    m = PktMat(3, 4)
    print(m.mat)
    print(m[0][1], type(m[0][1]))
    m[0][2] = 3
    print(m.mat)