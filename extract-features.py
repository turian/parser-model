#!/usr/bin/python

import common.hyperparameters, common.options
HYPERPARAMETERS = common.hyperparameters.read("attardi07_english_ptb")
common.options.reparse(HYPERPARAMETERS)

from common.file import myopen
import string

import common.featuremap

labelmap = common.featuremap.get(name="labels")
featuremap = common.featuremap.get(name="features")

#from collection import defaultdict
#cnt = defaultdict(int)
for l in myopen(HYPERPARAMETERS["original examples file"]):
    i = string.split(l)
    if len(i) == 0: continue
    l = i[0]
    feats = i[1:]
    labelmap.id(l, can_add=True)
    for f in feats:
        featuremap.id(f, can_add=True)
#        cnt[f] += 1

featuremap.readonly = True
labelmap.readonly = True

print "%d features" % featuremap.len
print "%d labels " % labelmap.len

featuremap.dump()
labelmap.dump()
