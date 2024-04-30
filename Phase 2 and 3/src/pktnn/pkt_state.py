from .pkt_fc import PktFc
from typing import List
import numpy as np


def save_state(fcs: List[PktFc], filename: str):
    weights = []
    for fc in fcs:
        weights.append(fc.weight.mat)
        weights.append(fc.bias.mat)
        weights.append(fc.dfa_weight.mat)
    np.savez(filename, *weights)


def load_state(fcs: List[PktFc], filename: str):
    weights = np.load(filename, allow_pickle=True)
    for i in range(len(fcs)):
        fcs[i].weight.mat = weights['arr_%d' % (i * 3)]
        fcs[i].bias.mat = weights['arr_%d' % (i * 3 + 1)]
        fcs[i].dfa_weight.mat = weights['arr_%d' % (i * 3 + 2)]
