#!/usr/bin/env python
#
#  Read in a DeSR features from stdin and include Brown features.
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

clusters = {}
for l in open(HYPERPARAMETERS["brown file"]):
    (cluster, word, count) = string.split(l)
    clusters[word] = cluster

from collections import defaultdict
missingcount = defaultdict(int)
totmissing = 0
tot = 0

for l in sys.stdin:
    for f in string.split(l):
        if f[:2] in ['B1', 'B0', '0B', '1B']:
            tot += 1
            trainw = f[2:]
            reprw = wordmapping.train_to_representation(trainw, clusters)
            if reprw in clusters:
                cluster = clusters[reprw]
                for p in HYPERPARAMETERS["brown prefix lengths"]:
                    print "%sBROWN%d_%s" % (f[:2], p, cluster[:p]),
            else:
                missingcount[trainw] += 1
                totmissing += 1
            if HYPERPARAMETERS["keep words"]: print f,
        else:
            print f,
    print

print >> sys.stderr, "%s word tokens have no representation" % percent(totmissing, tot)
print >> sys.stderr, "Most common missing words:", dictsort(missingcount)[:100]
