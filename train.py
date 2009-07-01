#!/usr/bin/env python

import common.hyperparameters, common.options
HYPERPARAMETERS = common.hyperparameters.read("attardi07_english_ptb")
common.options.reparse(HYPERPARAMETERS)

import examples
from vocabulary import *

import numpy as N

IDIM = featuremap.len
ODIM = labelmap.len
HID = HYPERPARAMETERS["hidden dimensions"]
LR = HYPERPARAMETERS["learning rate"]

from pylearn.algorithms.weights import random_weights
w1 = random_weights(IDIM, HID)
b1 = N.zeros(HID)
w2 = random_weights(HID, ODIM)
b2 = N.zeros(ODIM)

import graph

for (x, y) in examples.get_example():
#    print x, y
    print "Target y =", y
    o = graph.trainfn(x, N.array([y]), w1, b1, w2, b2)
    (kl, softmax, argmax, gw1, gb1, gw2, gb2) = o
    print "old KL=%.3f, softmax=%s, argmax=%d" % (kl, softmax, argmax)

    # Only sum the gradient along the non-zeroes.
    # How do we implement this as C code?
    for idx in x.indices:
        w1[idx,:] -= gw1[idx,:] * LR
#     w1 -= gw1 * LR
    w2 -= gw2 * LR
    b1 -= gb1 * LR
    b2 -= gb2 * LR

    o = graph.validatefn(x, N.array([y]), w1, b1, w2, b2)
    (kl, softmax, argmax) = o
    print "new KL=%.3f, softmax=%s, argmax=%d" % (kl, softmax, argmax)
