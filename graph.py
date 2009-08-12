
import common.hyperparameters, common.options
HYPERPARAMETERS = common.hyperparameters.read("attardi07_english_ptb")
HLAYERS = HYPERPARAMETERS["hidden layers"]
assert HLAYERS in [1, 2]

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

#import scipy.sparse as SS
#MTYPE = SS.csr_matrix
#TMTYPE = TS.csr_matrix
#_MTYPEs = [sparse.csc_matrix, sparse.csr_matrix, sparse.dok_matrix, sparse.lil_matrix, sparse.coo_matrix]
# * new class ``dia_matrix`` : the sparse DIAgonal format
# * new class ``bsr_matrix`` : the Block CS format

import theano.tensor.nnet as nnet
#ACTIVATION_FUNCTION = nnet.sigmoid
#import theano.tensor as t
#ACTIVATION_FUNCTION = t.tanh
from theano.sandbox.softsign import softsign
ACTIVATION_FUNCTION = softsign

x = TT.dmatrix()    # dvector?
#print targ1.shape
targety = TT.lvector()
#print x, targ

#random_weights
w1 = TT.dmatrix()
b1 = TT.dvector()
if HLAYERS == 2:
    wh = TT.dmatrix()
    bh = TT.dvector()
w2 = TT.dmatrix()
b2 = TT.dvector()

from theano.tensor.nnet import crossentropy_softmax_argmax_1hot_with_bias
from theano.compile.function_module import function

xw1 = theano.dot(w1.T, x.T).T
h = ACTIVATION_FUNCTION(xw1 + b1)

if HLAYERS == 2:
    xwh = theano.dot(wh.T, h.T).T
    h = ACTIVATION_FUNCTION(xwh + bh)

#zero = tensor.zeros_like(x[0,:])

if HYPERPARAMETERS["locally normalize"]:
    (kl, softmax, argmax) = crossentropy_softmax_argmax_1hot_with_bias(theano.dot(h, w2), b2, targety)
else:
    prey = theano.dot(h, w2) + b2
    softmax = nnet.sigmoid(prey)
    kl = -TT.mean(TT.sum(targety * TT.log(softmax) + (1 - targety) * TT.log(1 - softmax), axis=1), axis=0)
    argmax = TT.argmax(softmax)

if HLAYERS == 2:
    validatefn = function([x, targety, w1, b1, wh, bh, w2, b2], [kl, softmax, argmax, xw1, xwh], mode=COMPILE_MODE)
    (gw1, gb1, gwh, gbh, gw2, gb2) = TT.grad(kl, [w1, b1, wh, bh, w2, b2])
    trainfn = function([x, targety, w1, b1, wh, bh, w2, b2], [kl, softmax, argmax, xw1, xwh, theano.compile.io.Out(gw1, borrow = True), gb1, gwh, gbh, gw2, gb2], mode=COMPILE_MODE)
else:
    validatefn = function([x, targety, w1, b1, w2, b2], [kl, softmax, argmax, xw1], mode=COMPILE_MODE)
    (gw1, gb1, gw2, gb2) = TT.grad(kl, [w1, b1, w2, b2])
    trainfn = function([x, targety, w1, b1, w2, b2], [kl, softmax, argmax, xw1, theano.compile.io.Out(gw1, borrow = True), gb1, gw2, gb2], mode=COMPILE_MODE)
