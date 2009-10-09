#!/usr/bin/perl -w
#
#  Replace all colons ':' by 'COLON', so that colons can be used to
#  assign feature values.

while (<>) {
    s/:/COLON/g;
    print;
}
