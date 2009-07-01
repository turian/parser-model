#!/usr/bin/python

# TODO:
#   SPARSE
#       * Try csc, csr, lil, etc.
#       * Try true_dot, structured_dot, etc.
#       * Profile with minibatch
#       * Profile under different versions of scipy
#   * Different activation functions

DIM = 100000
HID = 5

# Learning rate
LR = 0.1

EPOCHS = 50

import theano.compile
COMPILE_MODE = theano.compile.Mode('c|py', 'fast_run')
#COMPILE_MODE = theano.compile.profilemode.ProfileMode(
#    theano.compile.mode.predefined_linkers['c|py'],
#    theano.compile.mode.predefined_optimizers['fast_run'])
#COMPILE_MODE = theano.compile.Mode('py', None)
#COMPILE_MODE = theano.compile.debugmode.DebugMode()

import theano
from theano import sparse as TS
from theano import tensor as TT

import scipy.sparse as SS
MTYPE = SS.csr_matrix
TMTYPE = TS.csr_matrix
#_MTYPEs = [sparse.csc_matrix, sparse.csr_matrix, sparse.dok_matrix, sparse.lil_matrix, sparse.coo_matrix]
# * new class ``dia_matrix`` : the sparse DIAgonal format
# * new class ``bsr_matrix`` : the Block CSR format

import theano.tensor.nnet as nnet
#ACTIVATION_FUNCTION = nnet.sigmoid
#import theano.tensor as t
#ACTIVATION_FUNCTION = t.tanh
from theano.sandbox.softsign import softsign
ACTIVATION_FUNCTION = softsign

import numpy as N
N.random.seed(0)

xvals = [
#[1., 2, 8, 9],
[1.],
[1, 2, 0, 4],
[1, 2, 6, 5],
[1, 2, 7, 8],
[8, 9],
[0, 4],
[6, 5],
[7, 8],
]
yvals = [[1.], [1], [1], [1], [0], [0], [0], [0]]
assert len(xvals) == len(yvals)
n = len(xvals)

xinstances = SS.lil_matrix((n, DIM))
for (i, ex) in enumerate(xvals):
    for j in ex:
        xinstances[i,j] = 1.
xinstances = MTYPE(xinstances)

targets = N.array(yvals)
targ1 = targets[0,:]

xR = TMTYPE('x')
#print targ1.shape
targR = TT.dvector("targ")
#print xR, targR

from pylearn.algorithms.weights import random_weights
w1 = random_weights(DIM, HID)
print "w1", w1, w1.shape, w1.dtype
b1 = N.zeros(HID)
print "b1", b1, b1.shape, b1.dtype
w2 = random_weights(HID, 1)
print "w2", w2, w2.shape, w2.dtype
b2 = N.zeros(1)
print "b2", b2, b2.shape, b2.dtype

#random_weights
w1R = TT.dmatrix('w1')
b1R = TT.dvector('b1')
w2R = TT.dmatrix('w2')
b2R = TT.dvector('b2')

import pylearn.algorithms.cost as cost
from theano.compile.function_module import function

#xw1R = theano.dot(w1R.T, xR.T).T
xw1R = TS.structured_dot(w1R.T, xR.T).T
#print w1R.type
#print xR.type

hR = ACTIVATION_FUNCTION(xw1R + b1R)
yR = nnet.sigmoid(theano.dot(hR, w2R).T + b2R)
loss = cost.KL_divergence(targR, yR)
fn = function([xR, targR, w1R, b1R, w2R, b2R], [yR, loss], mode=COMPILE_MODE)
(gw1, gb1, gw2, gb2) = TT.grad(loss, [w1R, b1R, w2R, b2R])
trainfn = function([xR, targR, w1R, b1R, w2R, b2R], [yR, loss, theano.compile.io.Out(gw1, borrow = True), gb1, gw2, gb2, hR], mode=COMPILE_MODE)
#print type(hR), type(yR)

print "TRAINING"
nex = xinstances.shape[0]
for epoch in range(EPOCHS):
    print "Epoch #", epoch
    for j in range(nex):
        #print "Example #", j
        x = xinstances[j,:]
#        #print "x", x.todense()
#        #print x.indices
        targety = targets[j,:]
        #print "target y", targety
        o = trainfn(x, targety, w1, b1, w2, b2)
#        for r in o:
           #print r, r.shape, r.dtype
        (predictedy, loss, gw1, gb1, gw2, gb2, h) = o
        #print "h", h
        #print "predicted y", predictedy
        #print "loss", loss
        #print gw1.shape, type(gw1), gw1.dtype
        # Only sum the gradient along the non-zeroes.
        # How do we implement this as C code?
        for idx in x.indices:
            w1[idx,:] -= gw1[idx,:] * LR
#        w1 -= gw1 * LR
        w2 -= gw2 * LR
        b1 -= gb1 * LR
        b2 -= gb2 * LR

#print "w1", w1
#print "b1", b1
#print "w2", w2
#print "b2", b2

#COMPILE_MODE.print_summary()


