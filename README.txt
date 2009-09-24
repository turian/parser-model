Parser model

    written by Joseph Turian

    available from: http://github.com/turian/parser-model/tree/master

A neural network with a sparse input, for predicting decisions of a natural
language syntax parser.

Designed to make decisions in Giuseppe Attardi's DESR parser, but usable in other
history-based parsers.
Little of this code is parser specific, it is mainly generic neural network
code.


This code relies upon Theano (http://pylearn.org/theano/)

==================================

DATA PREPROCESSING

You need a $DATA directory containing $DATA/english_ptb_train.feats

After editing the hyperparameters.*.yaml:

# Create files fmap.features.pkl.gz and fmap.labels.pkl.gz
./extract-features.py

# Create files $DATA/english_ptb_train.train.feats and $DATA/english_ptb_train.validation.feats
./partition-examples.py

# Shuffle examples
grep . $DATA/english_ptb_train.train.feats | ~/dev/common-scripts/shuffle.sh > $DATA/english_ptb_train.train.shuffled.feats
mv $DATA/english_ptb_train.train.feats $DATA/english_ptb_train.train.unshuffled.feats
ln -s english_ptb_train.train.shuffled.feats $DATA/english_ptb_train.train.feats
grep . $DATA/english_ptb_train.validation.feats | ~/dev/common-scripts/shuffle.sh > $DATA/english_ptb_train.validation.shuffled.feats
mv $DATA/english_ptb_train.validation.feats $DATA/english_ptb_train.validation.unshuffled.feats
ln -s english_ptb_train.validation.shuffled.feats $DATA/english_ptb_train.validation.feats
