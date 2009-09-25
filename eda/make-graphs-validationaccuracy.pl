#!/usr/bin/perl -w
#
#  Make a .dat file for each .err file.
#

$gnuplot = "plot";
$first = 1;
foreach $f (split(/[\r\n]+/, `ls [0-9]*err`)) {
    ($fnew = $f) =~ s/.err/-validationaccuracy.dat/;
    die $! if $fnew eq $f;
    print STDERR "$f => $fnew\n";
    $cmd = "cat $f | perl -ne 'print \"\$1 \$2\\n\" if m/After (\\S+) training examples, validation accuracy: (\\S+)%/;' > $fnew";
    print STDERR "$cmd\n";
    system($cmd);
    $gnuplot .= "," unless $first;
    $first = 0;
    $gnuplot .= " \\\n\t'$fnew' with lp"
}
print "$gnuplot\n";
