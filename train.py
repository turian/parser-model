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
import random

#from common import movingaverage
import common.movingaverage

random.seed(HYPERPARAMETERS["random seed"])
N.random.seed(HYPERPARAMETERS["random seed"])

IDIM = featuremap.len
ODIM = labelmap.len
HID = HYPERPARAMETERS["hidden dimensions"]
LR = HYPERPARAMETERS["learning rate"]
HLAYERS = HYPERPARAMETERS["hidden layers"]

from pylearn.algorithms.weights import random_weights
w1 = random_weights(IDIM, HID)
b1 = N.zeros(HID)
if HLAYERS == 2:
    wh = random_weights(HID, HID)
    bh = N.zeros(HID)
w2 = random_weights(HID, ODIM)
b2 = N.zeros(ODIM)

import graph
import state

def abs_prehidden(prehidden, str="Prehidden"):
    abs_prehidden = N.abs(prehidden)
    med = N.median(abs_prehidden)
    abs_prehidden = abs_prehidden.tolist()
    assert len(abs_prehidden) == 1
    abs_prehidden = abs_prehidden[0]
    abs_prehidden.sort()
    abs_prehidden.reverse()
    print >> sys.stderr, cnt, "Abs%s median =" % str, med, "max =", abs_prehidden[:5]

best_validation_accuracy = 0.
best_validation_at = 0
def validate():
    acc = []
    for (i, (x, y)) in enumerate(examples.get_validation_example()):
        if HYPERPARAMETERS["locally normalize"]:
            targety = N.array([y])
        else:
            targety = N.zeros(ODIM)
            targety[y] = 1.
        if HLAYERS == 2:
            o = graph.validatefn([x.data], targety, w1[x.indices], b1, wh, bh, w2, b2)
            (kl, softmax, argmax, prehidden1, prehidden2) = o
        else:
            o = graph.validatefn([x.data], targety, w1[x.indices], b1, w2, b2)
            (kl, softmax, argmax, prehidden) = o

        if argmax == y: acc.append(1.)
        else: acc.append(0.)

        if i < 5:
            if HLAYERS == 2:
                abs_prehidden(prehidden1, "Prehidden1")
                abs_prehidden(prehidden2, "Prehidden2")
            else:
                abs_prehidden(prehidden)       

    return N.mean(acc), N.std(acc)

mvgavg_accuracy = common.movingaverage.MovingAverage()
mvgavg_loss = common.movingaverage.MovingAverage()
cnt = 0
#if HLAYERS == 2:
#    state.save((w1, b1, wh, bh, w2, b2), rundir, best_validation_accuracy, best_validation_at)
#else:
#    state.save((w1, b1, w2, b2), rundir, best_validation_accuracy, best_validation_at)
for (x, y) in examples.get_training_example():
    cnt += 1
#    print x, y
#    print "Target y =", y

    if HYPERPARAMETERS["locally normalize"]:
        targety = N.array([y])
    else:
        targety = N.zeros(ODIM)
        targety[y] = 1.

    if HLAYERS == 2:
        o = graph.trainfn([x.data], targety, w1[x.indices], b1, wh, bh, w2, b2)
        (kl, softmax, argmax, prehidden1, prehidden2, gw1, gb1, gwh, gbh, gw2, gb2) = o
    else:
        o = graph.trainfn([x.data], targety, w1[x.indices], b1, w2, b2)
        (kl, softmax, argmax, prehidden, gw1, gb1, gw2, gb2) = o
#    print "old KL=%.3f, softmax=%s, argmax=%d" % (kl, softmax, argmax)
#    print "old KL=%.3f, argmax=%d" % (kl, argmax)

    if argmax == y: this_accuracy = 100.
    else: this_accuracy = 0.
    mvgavg_accuracy.add(this_accuracy)

    if N.isnan(kl).any():
        print >> sys.stderr, "Got a NaN loss"
    else:
        mvgavg_loss.add(kl)
    
        # Only sum the gradient along the non-zeroes.
        # How do we implement this as C code?
    #    for idx in x.indices:
    #        w1[idx,:] -= gw1[idx,:] * LR
        w1[x.indices] -= gw1 * LR
    #     w1 -= gw1 * LR
        b1 -= gb1 * LR
        if HLAYERS == 2:
            wh -= gwh * LR
            bh -= gbh * LR
        w2 -= gw2 * LR
        b2 -= gb2 * LR
    
    #    o = graph.validatefn(x, N.array([y]), w1, b1, w2, b2)
    #    (kl, softmax, argmax, presquashh) = o
    ##    print "new KL=%.3f, softmax=%s, argmax=%d" % (kl, softmax, argmax)
    #    print "new KL=%.3f, argmax=%d" % (kl, argmax)
    
    if cnt % HYPERPARAMETERS["examples per validation"] == 0:
        valacc, valstd = validate()
        sys.stderr.write("After %d training examples, validation accuracy: %.2f%%, stddev: %.2f%% (former best=%.2f%% at %d)\n" % (cnt, valacc*100, valstd*100, best_validation_accuracy*100, best_validation_at))
        if best_validation_accuracy < valacc:
            best_validation_accuracy = valacc
            best_validation_at = cnt
            sys.stderr.write("NEW BEST VALIDATION ACCURACY. Saving state.\n")
            if HLAYERS == 2:
                state.save((w1, b1, wh, bh, w2, b2), rundir, best_validation_accuracy, best_validation_at)
            else:
                state.save((w1, b1, w2, b2), rundir, best_validation_accuracy, best_validation_at)
        elif cnt > 2*best_validation_at and cnt >= HYPERPARAMETERS["minimum training updates"]:
            sys.stderr.write("Have not beaten best validation accuracy for a while. Terminating training...\n")
            sys.stderr.write(stats() + "\n")
            break
    if cnt % 1000 == 0:
        sys.stderr.write("After %d training examples, training accuracy %s\n" % (cnt, mvgavg_accuracy))
        sys.stderr.write("After %d training examples, training loss %s\n" % (cnt, mvgavg_loss))
        sys.stderr.write(stats() + "\n")

#graph.COMPILE_MODE.print_summary()
