#!/usr/bin/env python
#
#  Read in a DeSR features from stdin and include embeddings features.
#  We assume that case will remain intact.
#

import sys, string, re
import common.hyperparameters, common.options
HYPERPARAMETERS = common.hyperparameters.read("features")
common.options.reparse(HYPERPARAMETERS)
print >> sys.stderr, HYPERPARAMETERS

from common.str import percent
from common.mydict import sort as dictsort
import wordmapping

print >> sys.stderr, "Reading embeddings from %s..." % HYPERPARAMETERS["embeddings file"]
embeddings = {}
for l in open(HYPERPARAMETERS["embeddings file"]):
    lst = string.split(l)
    word = lst[0]
    e = [float(f) for f in lst[1:]]
    assert len(e) == HYPERPARAMETERS["embeddings length"]
    embeddings[word] = e
print >> sys.stderr, "...done reading embeddings from %s" % HYPERPARAMETERS["embeddings file"]

from collections import defaultdict
missingcount = defaultdict(int)
totmissing = 0
tot = 0

for l in sys.stdin:
    for f in string.split(l):
        if f[:2] in ['B1', 'B0', '0B', '1B']:
            tot += 1
            trainw = f[2:]
            reprw = wordmapping.train_to_representation(trainw, embeddings)
            if reprw in embeddings:
                continue
            elif HYPERPARAMETERS["embeddings unknown word"] in embeddings:
                reprw = HYPERPARAMETERS["embeddings unknown word"]
                missingcount[trainw] += 1
                totmissing += 1
            else:
                reprw = None
                missingcount[trainw] += 1
                totmissing += 1
            if reprw is not None:
                e = embeddings[reprw]
                for p in range(HYPERPARAMETERS["embeddings length"]):
                    print "%sEMBED%d:%f" % (f[:2], p, e[p]),
            if HYPERPARAMETERS["keep words"]: print f,
        else:
            print f,
    print

print >> sys.stderr, "%s word tokens have no representation" % percent(totmissing, tot)
print >> sys.stderr, "Most common missing words:", dictsort(missingcount)[:100]
