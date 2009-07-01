#!/usr/bin/env python

import common.hyperparameters, common.options
HYPERPARAMETERS = common.hyperparameters.read("attardi07_english_ptb")
common.options.reparse(HYPERPARAMETERS)

import common.dump
rundir = common.dump.create_canonical_directory(HYPERPARAMETERS)

import examples
from vocabulary import *
from common.stats import stats
from common.file import myopen
import sys
import numpy as N
import math
from os.path import join
import cPickle

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

best_validation_accuracy = 0.
def validate():
    acc = []
    for (x, y) in examples.get_validation_example():
        o = graph.validatefn(x, N.array([y]), w1, b1, w2, b2)
        if argmax == y: acc.append(1.)
        else: acc.append(0.)
    return N.mean(acc), N.std(acc)

def state_save():
    cPickle.dump((w1, b1, w2, b2), myopen(join(rundir, "best-model.pkl"), "w"), protocol=-1)
    myopen(join(rundir, "best-model-validation-accuracy.txt"), "w").write("%.2f%%" % (best_validation_accuracy*100))

mvgavg_accuracy = 0.
mvgavg_variance = 0.
cnt = 0
for (x, y) in examples.get_training_example():
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

    if cnt % HYPERPARAMETERS["examples per validation"] == 0:
        valacc, valstd = validate()
        sys.stderr.write("After %d training examples, validation accuracy: %.2f%%, stddev: %.2f%%\n" % (cnt, valacc*100, valstd*100))
        if best_validation_accuracy < valacc:
            best_validation_accuracy = valacc
            sys.stderr.write("NEW BEST VALIDATION ACCURACY. Saving state.\n")
            state_save()
    if cnt % 1000 == 0:
        sys.stderr.write("After %d training examples, training accuracy (moving average): %.2f%%, stddev: %.2f%%\n" % (cnt, 100. * mvgavg_accuracy, 100. * math.sqrt(mvgavg_variance)))
        sys.stderr.write(stats() + "\n")

#graph.COMPILE_MODE.print_summary()
