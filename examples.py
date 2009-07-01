"""
Read the training examples.
"""

import string
import common.file
import common.hyperparameters
import common.scipysparse as sparse
import common.featuremap
from vocabulary import *

def get_example():
    """
    Get an example, as an infinite loop.
    """
    HYPERPARAMETERS = common.hyperparameters.read("attardi07_english_ptb")
    while 1:
        for l in common.file.myopen(HYPERPARAMETERS["examples file"]):
            i = string.split(l)
            if len(i) == 0: continue
            l = i[0]
            feats = i[1:]
            x = sparse.lil_matrix((1, featuremap.len))
            for f in feats:
                x[0, featuremap.id(f)] = 1.
            x = sparse.csr_matrix(x)

            y = labelmap.id(l)
            yield x, y

#for (x, y) in get_example():
#    print x, y
