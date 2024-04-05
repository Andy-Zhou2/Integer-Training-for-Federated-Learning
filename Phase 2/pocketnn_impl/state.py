from pktnn_fc import PktFc
from pktnn_mat import PktMat
from typing import List
import numpy as np


def save_state(fcs: List[PktFc], filename: str):
    weights = []
    for fc in fcs:
        weights.append(fc.weight)
        weights.append(fc.bias)
    np.savez(filename, *weights)


def load_state(fcs: List[PktFc], filename: str):
    weights = np.load(filename)
    for i in range(len(fcs)):
        fcs[i].weight = weights['arr_%d' % (i * 2)]
        fcs[i].bias = weights['arr_%d' % (i * 2 + 1)]
