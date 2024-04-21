import numpy as np
from ..utils.utils_calc import truncate_divide


def rand_range(lower: int, upper: int):
    r = np.random.randint(256 ** 4, dtype='<u4', size=1)[0]
    return (r % (upper - lower + 1)) + lower


def mat_mul_mat(a, b):
    result = PktMat(a.row, b.col)
    result.mat = np.matmul(a.mat, b.mat)
    return result


def mat_add_mat(a, b):
    result = PktMat(a.row, a.col)
    result.mat = np.add(a.mat, b.mat)
    return result


def transpose_of(other):
    result = PktMat(other.col, other.row)
    result.mat = np.transpose(other.mat)
    return result


def mat_div_const(a, const):
    result = PktMat(a.row, a.col)
    result.mat = truncate_divide(a.mat, const)
    return result


def mat_elem_div_mat(a: 'PktMat', b: 'PktMat'):
    result = PktMat(a.row, a.col)
    result.mat = truncate_divide(a.mat, b.mat)
    return result


def mat_elem_mul_mat(a, b):
    result = PktMat(a.row, a.col)
    result.mat = np.multiply(a.mat, b.mat)
    return result


def mat_mul_const(a, const):
    result = PktMat(a.row, a.col)
    result.mat = np.multiply(a.mat, const)
    return result


def deep_copy(pkt_mat: 'PktMat'):
    result = PktMat(pkt_mat.row, pkt_mat.col)
    result.mat = pkt_mat.mat.copy()
    return result


class PktMat:
    def init_zeros(self, row, col):
        self.row = row
        self.col = col
        self.mat = np.zeros((row, col), dtype=np.int64)

    @property
    def shape(self):
        return self.row, self.col

    def __repr__(self):
        return f'PktMat({self.row}, {self.col})'

    @classmethod
    def fill(cls, row, col, value):
        pkt_mat = cls(row, col)
        pkt_mat.mat.fill(value)
        return pkt_mat

    def __init__(self, row=0, col=0, mat=None):
        self.mat = None
        self.init_zeros(row, col)  # self.mat initialized to zeros

        if mat is not None:
            assert row == mat.shape[0] and col == mat.shape[1]
            assert mat.dtype == np.int64
            self.mat = mat

    def __getitem__(self, key):
        # does not need __setitem__ because we can use self.mat[key] = value
        # it returns a numpy reference
        return self.mat[key]

    def __setitem__(self, key, value):
        # it returns a numpy reference
        self.mat[key] = value

    def get_max_index_in_row(self, row):
        return self.mat[row].argmax()

    def self_add_mat(self, b):
        self.mat = np.add(self.mat, b.mat)

    def dims_equal(self, other):
        return self.row == other.row and self.col == other.col

    def self_div_const(self, const):
        self.mat = truncate_divide(self.mat, const)

    def reset_all(self, row, col, value):
        self.init_zeros(row, col)
        self.mat.fill(value)

    def clamp_mat(self, min_val, max_val):
        self.mat = np.clip(self.mat, min_val, max_val)

    def self_elem_div_mat(self, b: 'PktMat'):
        self.mat = truncate_divide(self.mat, b.mat)

    def self_mul_const(self, const: int):
        self.mat = np.multiply(self.mat, const)

    def mat_elem_mul_self(self, other: 'PktMat'):
        temp = deep_copy(self)
        self.mat = mat_elem_mul_mat(other, temp).mat

    def set_random(self, allow_zero: bool, min_val: int, max_val: int):
        self.mat = np.random.randint(min_val, max_val, size=(self.row, self.col), dtype=np.int64)
        if not allow_zero:
            self.mat += (self.mat == 0)

        # to ensure reproducibility which agrees with C++: (considering the seeding at the top of the file)
        # for r in range(self.row):
        #     for c in range(self.col):
        #         self.mat[r][c] = rand_range(min_val, max_val)
        #         if not allow_zero:
        #             while self.mat[r][c] == 0:
        #                 self.mat[r][c] = rand_range(min_val, max_val)

    def hash(self):
        def simple_hash(string):
            hash = 5381  # Starting with a large prime number as a seed
            for c in string:
                hash = (((hash << 5) + hash) + ord(c)) % 1000000007  # hash * 33 + ord(c)
            return hash % 1000000007  # Use a large prime number to wrap around

        matrix_str = ''.join([''.join([str(item) for item in row]) for row in self.mat])
        return simple_hash(matrix_str)

    def sum(self):
        return np.sum(self.mat)

    def print(self):
        for i in range(self.row):
            for j in range(self.col):
                print(self.mat[i][j], end='\t')
            print()


if __name__ == '__main__':
    m = PktMat(3, 4)
    print(m.mat)
    print(m[0][1], type(m[0][1]))
    m[0][2] = 3
    print(m.mat)

    m[0:2] = 5
    print(m.mat)
