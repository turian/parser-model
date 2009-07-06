
from common.file import myopen
from os.path import join
import cPickle
import sys
from common.stats import stats

def _filename(rundir):
    return join(rundir, "best-model.pkl")

def save():
    import common.hyperparameters
    HYPERPARAMETERS = common.hyperparameters.read("attardi07_english_ptb")
    HLAYERS = HYPERPARAMETERS["hidden layers"]
    if HLAYERS == 2:
        cPickle.dump((w1, b1, wh, bh, w2, b2), myopen(_filename(rundir), "w"), protocol=-1)
    else:
        cPickle.dump((w1, b1, w2, b2), myopen(_filename(rundir), "w"), protocol=-1)
    myopen(join(rundir, "best-model-validation.txt"), "w").write("Accuracy %.2f%% after %d updates" % (best_validation_accuracy*100, best_validation_at))

def load(rundir):
    print >> sys.stderr, "Loading state from %s..." % _filename(rundir)
    print >> sys.stderr, stats()
    m = cPickle.load(myopen(_filename(rundir), "r"))
    print >> sys.stderr, "...done loading state from %s" % _filename(rundir)
    print >> sys.stderr, stats()
    return m
