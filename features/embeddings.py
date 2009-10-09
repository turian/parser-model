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

print >> sys.stderr, "Reading embeddings from %s..." % HYPERPARAMETERS["embeddings file"]
embeddings = {}
for l in open(HYPERPARAMETERS["embeddings file"]):
    lst = string.split(l)
    word = lst[0]
    e = [float(f) for f in lst[1:]]
    assert len(e) == HYPERPARAMETERS["embeddings length"]
    embeddings[word] = e
print >> sys.stderr, "...done reading embeddings from %s" % HYPERPARAMETERS["embeddings file"]

for l in sys.stdin:
    for f in string.split(l):
        if f[:2] in ['B1', 'B0', '0B', '1B']:
            if f[2:] in embeddings:
                w = f[2:]
            elif f[2:] == "COLON" and ":" in embeddings:
                w = ":"
            elif HYPERPARAMETERS["embeddings unknown word"] in embeddings:
                w = HYPERPARAMETERS["embeddings unknown word"]
            else:
                w = None
            if w is not None:
                e = embeddings[w]
                for p in range(HYPERPARAMETERS["embeddings length"]):
                    print "%sEMBED%d:%f" % (f[:2], p, e[p]),
            if HYPERPARAMETERS["keep words"]: print f,
        else:
            print f,
    print
