#!/usr/bin/env python

import common.hyperparameters, common.options
HYPERPARAMETERS = common.hyperparameters.read("attardi07_english_ptb")
common.options.reparse(HYPERPARAMETERS)

import examples
from vocabulary import *
from common.stats import stats
import sys
import numpy as N
import math

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

mvgavg_accuracy = 0.
mvgavg_variance = 0.
cnt = 0
for (x, y) in examples.get_example():
    cnt += 1
#    print x, y
#    print "Target y =", y
    o = graph.trainfn(x, N.array([y]), w1, b1, w2, b2)
    (kl, softmax, argmax, gw1, gb1, gw2, gb2) = o
#    print "old KL=%.3f, softmax=%s, argmax=%d" % (kl, softmax, argmax)
#    print "old KL=%.3f, argmax=%d" % (kl, argmax)

    if argmax == y: this_accuracy = 1.
    else: this_accuracy = 0.
    mvgavg_accuracy = mvgavg_accuracy - (2. / cnt) * (mvgavg_accuracy - this_accuracy)
    # Should I compute mvgavg_variance before updating the mvgavg_accuracy?
    this_variance = (this_accuracy - mvgavg_accuracy) * (this_accuracy - mvgavg_accuracy)
    mvgavg_variance = mvgavg_variance - (2. / cnt) * (mvgavg_variance - this_variance)
#    print "Accuracy (moving average): %.2f%%, stddev: %.2f%%" % (100. * mvgavg_accuracy, 100. * math.sqrt(mvgavg_variance))

    # Only sum the gradient along the non-zeroes.
    # How do we implement this as C code?
    for idx in x.indices:
        w1[idx,:] -= gw1[idx,:] * LR
#     w1 -= gw1 * LR
    w2 -= gw2 * LR
    b1 -= gb1 * LR
    b2 -= gb2 * LR

#    o = graph.validatefn(x, N.array([y]), w1, b1, w2, b2)
#    (kl, softmax, argmax) = o
##    print "new KL=%.3f, softmax=%s, argmax=%d" % (kl, softmax, argmax)
#    print "new KL=%.3f, argmax=%d" % (kl, argmax)

    if cnt % 100 == 0:
        sys.stderr.write("After %d training examples, accuracy (moving average): %.2f%%, stddev: %.2f%%\n" % (cnt, 100. * mvgavg_accuracy, 100. * math.sqrt(mvgavg_variance)))
        sys.stderr.write(stats() + "\n")
