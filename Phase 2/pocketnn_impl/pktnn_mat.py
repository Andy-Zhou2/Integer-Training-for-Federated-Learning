import numpy as np


class PktMat:
    def init_zeros(self, row, col):
        self.row = row
        self.col = col
        self.mat = np.zeros((row, col), dtype=np.int64)

    @property
    def shape(self):
        return self.row, self.col

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

    def mat_mul_mat(self, a, b):
        self.init_zeros(a.row, b.col)
        self.mat = np.matmul(a.mat, b.mat)

    def mat_add_mat(self, a, b):
        self.mat = np.add(a.mat, b.mat)

    def self_add_mat(self, b):
        self.mat_add_mat(self, b)
        return self

    def dims_equal(self, other):
        return self.row == other.row and self.col == other.col

    def transpose_of(self, other):
        self.init_zeros(other.col, other.row)
        self.mat = np.transpose(other.mat)
        return self

    def self_div_const(self, const):
        self.mat = np.divide(self.mat, const)
        return self

    def mat_div_const(self, a, const):
        self.init_zeros(a.row, a.col)
        self.mat = np.divide(a.mat, const)
        return self

    def reset_all(self, row, col, value):
        self.init_zeros(row, col)
        self.mat.fill(value)
        return self

    def clamp_mat(self, min, max):
        self.mat = np.clip(self.mat, min, max)
        return self

    def mat_elem_div_mat(self, a, b):
        self.init_zeros(a.row, a.col)
        self.mat = np.divide(a.mat, b.mat)
        return self

    def self_elem_div_mat(self, b):
        self.mat = np.divide(self.mat, b.mat)
        return self

    def mat_elem_mul_mat(self, a, b):
        self.init_zeros(a.row, a.col)
        self.mat = np.multiply(a.mat, b.mat)
        return self

    def self_mul_const(self, const):
        self.mat = np.multiply(self.mat, const)
        return self

    def mat_mul_const(self, a, const):
        self.init_zeros(a.row, a.col)
        self.mat = np.multiply(a.mat, const)
        return self

    def deep_copy_of(self, other):
        self.init_zeros(other.row, other.col)
        self.mat = other.mat.copy()
        return self
    def mat_elem_mul_self(self, other):
        temp = self.deep_copy_of(self)
        self.mat_elem_mul_mat(other, temp)
        return self

    def set_random(self, allow_zero: bool, min_val: int, max_val: int):
        self.mat = np.random.randint(min_val, max_val, size=(self.row, self.col), dtype=np.int64)
        if not allow_zero:
            self.mat += (self.mat == 0)

if __name__ == '__main__':
    m = PktMat(3, 4)
    print(m.mat)
    print(m[0][1], type(m[0][1]))
    m[0][2] = 3
    print(m.mat)

    m[0:2] = 5
    print(m.mat)