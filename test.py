#!/usr/bin/env python

import state

if __name__ == "__main__":
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-m", "--modeldir", dest="modeldir", type="string", help="directory from which to load model")
    (options, args) = parser.parse_args()
    assert len(args) == 0
    assert options.modeldir is not None

    import common.dump, yaml
    parameters = common.dump.load_canonical_directory(options.modeldir)
    import common.hyperparameters
    common.hyperparameters.set(parameters, "attardi07_english_ptb")
    HYPERPARAMETERS = common.hyperparameters.read("attardi07_english_ptb")

    m = state.load(options.modeldir)

    HLAYERS = HYPERPARAMETERS["hidden layers"]
    if HLAYERS == 2:
        w1, b1, wh, bh, w2, b2 = m
    else:
        w1, b1, w2, b2 = m

    import examples, sys
    import graph
    import numpy as N
    from vocabulary import labelmap
    from common.mydict import sort as dictsort
    for l in sys.stdin:
        e = examples._example_from_string(l)
        (x, y) = e
        if HLAYERS == 2:
            o = graph.validatefn([x.data], N.array([y]), w1[x.indices], b1, wh, bh, w2, b2)
            (kl, softmax, argmax, prehidden1, prehidden2) = o
        else:
            o = graph.validatefn([x.data], N.array([y]), w1[x.indices], b1, w2, b2)
            (kl, softmax, argmax, prehidden) = o

        assert softmax.shape[0] == 1
        softmax = softmax[0]
        prs = {}
        for i in range(softmax.shape[0]):
            prs[labelmap.str(i)] = softmax[i]
        print dictsort(prs)[:3]
#        print argmax, softmax
