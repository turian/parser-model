"""
Read the training examples.
"""

import string, sys
import common.file
import common.hyperparameters
import common.scipysparse as sparse
import common.featuremap
from common.stats import stats
from vocabulary import *

def _example_from_string(l):
    i = string.split(l)
    l = i[0]
    feats = i[1:]
    x = sparse.lil_matrix((1, featuremap.len))
    for f in feats:
        x[0, featuremap.id(f)] = 1.
    x = sparse.csr_matrix(x)
    y = labelmap.id(l)
    return x, y

def get_training_example():
    """
    Get a training example, as an infinite loop.
    """
    HYPERPARAMETERS = common.hyperparameters.read("attardi07_english_ptb")
    epoch = 0
    examples = 0
    while 1:
        epoch += 1
        sys.stderr.write("STARTING EPOCH #%d (%d examples)\n" % (epoch, examples))
        sys.stderr.write(stats() + "\n")
        for l in common.file.myopen(HYPERPARAMETERS["train examples file"]):
            if l == "\n": continue
            examples += 1
            yield _example_from_string(l)

def get_validation_example():
    """
    Get an validation example, one pass.
    """
    HYPERPARAMETERS = common.hyperparameters.read("attardi07_english_ptb")
    for l in common.file.myopen(HYPERPARAMETERS["validation examples file"]):
        if l == "\n": continue
        yield _example_from_string(l)

#for (x, y) in get_example():
#    print x, y
