#!/bin/bash
# Independent cross-check: the toolbox-free MATLAB mirror run in GNU Octave for
# all three feeders and both DER cases (10 replications each, no sweep).
cd "$(dirname "$0")"
: > octave_crosscheck_log.txt
for f in ieee33 ieee69 ieee123; do
  for c in moderate der_rich; do
    octave --no-gui --quiet --eval "main_experiment('$c', 10, false, '$f');" >> octave_crosscheck_log.txt 2>&1
  done
done
echo OCTAVE_DONE >> octave_crosscheck_log.txt
