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

clusters = {}
for l in open(HYPERPARAMETERS["brown file"]):
    (cluster, word, count) = string.split(l)
    clusters[word] = cluster

for l in sys.stdin:
    for f in string.split(l):
        if f[:2] in ['B1', 'B0', '0B', '1B']:
            if f[2:] in clusters:
                cluster = clusters[f[2:]]
                for p in HYPERPARAMETERS["brown prefix lengths"]:
                    print "%sBROWN%d_%s" % (f[:2], p, cluster[:p]),
            if HYPERPARAMETERS["keep words"]: print f,
        else:
            print f,
    print
