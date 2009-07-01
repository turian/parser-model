After editing the hyperparameters.*.yaml:

./extract-features.py

./partition-examples.py

# Shuffle examples
grep . data/english_ptb_train.train.feats | ~/dev/common-scripts/shuffle.sh > data/english_ptb_train.train.shuffled.feats
mv data/english_ptb_train.train.feats data/english_ptb_train.train.unshuffled.feats
ln -s data/english_ptb_train.train.shuffled.feats data/english_ptb_train.train.feats
grep . data/english_ptb_train.validation.feats | ~/dev/common-scripts/shuffle.sh > data/english_ptb_train.validation.shuffled.feats
mv data/english_ptb_train.validation.feats data/english_ptb_train.validation.unshuffled.feats
ln -s data/english_ptb_train.validation.shuffled.feats data/english_ptb_train.validation.feats
