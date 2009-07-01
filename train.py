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

from pylearn.algorithms.weights import random_weights
w1 = random_weights(IDIM, HID)
b1 = N.zeros(HID)
w2 = random_weights(HID, ODIM)
b2 = N.zeros(ODIM)

import graph

for (x, y) in examples.get_example():
#    print x, y
    print "Target y =", y
    (kl, softmax, argmax) = graph.validatefn(x, N.array([y]), w1, b1, w2, b2)
    print "KL=%d, softmax=%s, argmax=%d" % (kl, softmax, argmax)

#print "TRAINING"
#nex = xinstances.shape[0]
#for epoch in range(EPOCHS):
#    print "Epoch #", epoch
#    for j in range(nex):
#        #print "Example #", j
#        x = xinstances[j,:]
##        #print "x", x.todense()
##        #print x.indices
#        targety = targets[j,:]
#        #print "target y", targety
#        o = trainfn(x, targety, w1, b1, w2, b2)
##        for r in o:
#           #print r, r.shape, r.dtype
#        (predictedy, loss, gw1, gb1, gw2, gb2, h) = o
#        #print "h", h
#        #print "predicted y", predictedy
#        #print "loss", loss
#        #print gw1.shape, type(gw1), gw1.dtype
#        # Only sum the gradient along the non-zeroes.
#        # How do we implement this as C code?
#        for idx in x.indices:
#            w1[idx,:] -= gw1[idx,:] * LR
##        w1 -= gw1 * LR
#        w2 -= gw2 * LR
#        b1 -= gb1 * LR
#        b2 -= gb2 * LR
#
##print "w1", w1
##print "b1", b1
##print "w2", w2
##print "b2", b2
#
##COMPILE_MODE.print_summary()
