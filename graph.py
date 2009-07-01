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

xR = TMTYPE('x')
#print targ1.shape
targetyR = TT.lvector()
#print xR, targR

#random_weights
w1R = TT.dmatrix('w1')
b1R = TT.dvector('b1')
w2R = TT.dmatrix('w2')
b2R = TT.dvector('b2')

from theano.tensor.nnet import crossentropy_softmax_argmax_1hot_with_bias
from theano.compile.function_module import function

#xw1R = theano.dot(w1R.T, xR.T).T
xw1R = TS.structured_dot(w1R.T, xR.T).T
#print w1R.type
#print xR.type

hR = ACTIVATION_FUNCTION(xw1R + b1R)

#zero = tensor.zeros_like(x[0,:])
(kl, softmax, argmax) = crossentropy_softmax_argmax_1hot_with_bias(theano.dot(hR, w2R), b2R, targetyR)

validatefn = function([xR, targetyR, w1R, b1R, w2R, b2R], [kl, softmax, argmax], mode=COMPILE_MODE)
(gw1, gb1, gw2, gb2) = TT.grad(kl, [w1R, b1R, w2R, b2R])
trainfn = function([xR, targetyR, w1R, b1R, w2R, b2R], [kl, softmax, argmax, theano.compile.io.Out(gw1, borrow = True), gb1, gw2, gb2], mode=COMPILE_MODE)
