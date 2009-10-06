Parser model

    written by Joseph Turian

    available from: http://github.com/turian/parser-model/tree/master

A neural network with a sparse input, for predicting decisions of a natural
language syntax parser.

Designed to make decisions in Giuseppe Attardi's DeSR parser, but usable in other
history-based parsers.
Little of this code is parser specific, it is mainly generic neural network
code.

desr.txt is a notepad of my results while experimenting with the DeSR
parser.


This code relies upon Theano (http://pylearn.org/theano/)


After editing the hyperparameters.*.yaml:

./extract-features.py

./partition-examples.py

# Shuffle examples
grep . data/english_ptb_train.train.feats | ~/dev/common-scripts/shuffle.sh > data/english_ptb_train.train.shuffled.feats
mv data/english_ptb_train.train.feats data/english_ptb_train.train.unshuffled.feats
ln -s english_ptb_train.train.shuffled.feats data/english_ptb_train.train.feats
grep . data/english_ptb_train.validation.feats | ~/dev/common-scripts/shuffle.sh > data/english_ptb_train.validation.shuffled.feats
mv data/english_ptb_train.validation.feats data/english_ptb_train.validation.unshuffled.feats
ln -s english_ptb_train.validation.shuffled.feats data/english_ptb_train.validation.feats
