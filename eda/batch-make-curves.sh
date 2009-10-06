#!/bin/sh

# Make all dat files
../../eda/make-graphs-validationaccuracy.pl

# Sort all dat files
# First perl recipe adds gnuplot codes
# Second perl recipe strips final ', \'  to prevent gnuplot error
echo > graphs-parser-model-validationaccuracy.gp
echo "set terminal postscript color 12" >> graphs-parser-model-validationaccuracy.gp
echo "set output 'graphs-parser-model-validationaccuracy.ps'" >> graphs-parser-model-validationaccuracy.gp
echo "set key bottom" >> graphs-parser-model-validationaccuracy.gp
#echo "set logscale y" >> graphs-parser-model-validationaccuracy.gp
echo "plot [] [90:] \\" >> graphs-parser-model-validationaccuracy.gp
~/dev/common-scripts/sort-curves.py *validationaccuracy.dat | perl -ne "chop; print \"\\t'\$_' with lp, \\\\\\n\"" | perl -e '$str = ""; while(<>){ $str .= $_; } $str =~ s/, \\$//s; print $str' >> graphs-parser-model-validationaccuracy.gp

gnuplot graphs-parser-model-validationaccuracy.gp
ps2pdf graphs-parser-model-validationaccuracy.ps
cp *pdf ~/public_html/priv ; chmod a+r ~/public_html/priv/*pdf
