"""
Mapping between words in the training space and words in the representation data.
We leave case intact, but here would be where to change it.
"""

import re
import sys

colonre = re.compile("COLON")
slashre = re.compile('\\\/')
def train_to_representation(trainw, representations):
    reprw = trainw
    reprw = colonre.sub(":", reprw)
    reprw = slashre.sub('/', reprw)
    if reprw == "``": reprw = '"'
    if reprw == "''": reprw = '"'
    if reprw == "`": reprw = "'" 

    # Remove final punctuation, if it will make a word have a representation
    if len(reprw) > 1 and reprw[-1] == '.' and reprw not in representations and reprw[:-1] in representations:
#        print >> sys.stderr, reprw,
        reprw = reprw[:-1]
#        print >> sys.stderr, reprw

#    if reprw != trainw and reprw != '"': print >> sys.stderr, trainw, reprw

    return reprw
