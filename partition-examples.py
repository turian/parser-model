#!/usr/bin/python
#
#  Partition examples into train and validation
#

import random

import common.hyperparameters, common.options
HYPERPARAMETERS = common.hyperparameters.read("attardi07_english_ptb")
common.options.reparse(HYPERPARAMETERS)

random.seed(HYPERPARAMETERS["random seed"])

from common.file import myopen
import string

Tf = HYPERPARAMETERS["train examples file"]
Vf = HYPERPARAMETERS["validation examples file"]
T = open(Tf, "wt")
V = open(Vf, "wt")

print "Writing to %s and %s" % (Tf, Vf)

ex = ""
for l in myopen(HYPERPARAMETERS["original examples file"]):
    ex += l
    if string.strip(l) == "":
        if random.random() < HYPERPARAMETERS["validation example likelihood"]:
            V.write(ex)
        else:
            T.write(ex)
        ex = ""

if ex != "":
    if random.random() < HYPERPARAMETERS["validation example likelihood"]:
        V.write(ex)
    else:
        T.write(ex)
