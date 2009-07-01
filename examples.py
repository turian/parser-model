"""
Read the training examples.
"""

import string
import common.file
import common.hyperparameters
import common.featuremap
labelmap = common.featuremap.get(name="labels")
featuremap = common.featuremap.get(name="features")
labelmap.readonly = True
featuremap.readonly = True

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
            x = frozenset([featuremap.id(f) for f in feats])
            y = labelmap.id(l)
            yield x, y

#for (x, y) in get_example():
#    print x, y
