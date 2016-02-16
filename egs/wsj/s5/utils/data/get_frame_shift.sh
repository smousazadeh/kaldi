#!/bin/bash

# Copyright 2016  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

# This script takes as input a data directory, such as data/train/, preferably
# with utt2dur file already existing (or the utt2dur file will be created if
# not), and it attempts to work out the frame shift by comparing the utt2dur
# with the output of feat-to-len on the feats.scp.  It prints it out.  If the
# computed frame shift does not seem to be a multiple of 0.005 seconds, the script
# will die.  Any errors will be printed to the standard error.

. utils/parse_options.sh
. ./path.sh

if [ $# != 1 ]; then
  echo "Usage: $0 <datadir>"
  echo "e.g.:"
  echo " $0 data/train"
  echo "This script prints the frame-shift (e.g. 0.01) to the standard out."
  echo "If <datadir> does not contain utt2dur, this script will call utils/data/get_utt2dur.sh,"
  echo "which will require write permission to <datadir>"
  exit 1
fi

export LC_ALL=C

dir=$1

if [ ! -f $dir/utt2dur ]; then
  echo "$0: $dir/utt2dur does not exist: creating it" 1>&2
  utils/data/get_utt2dur.sh $dir 1>&2
fi

if [ ! -f $dir/feats.scp ]; then
  echo "$0: $dir/feats.scp does not exist" 1>&2
  exit 1
fi

temp=$(mktemp /tmp/tmp.XXXX)

feat-to-len scp:$dir/feats.scp ark,t:- | head -n 10 > $temp

if [ -z $temp ]; then
  echo "$0: error running feat-to-len" 1>&2
  exit 1
fi

head -n 10 $dir/utt2dur | paste - $temp | \
   awk '{ dur += $2; frames += $4; } END { shift = dur / frames;
       shift_scaled = shift / 0.005; int_shift_scaled = int(0.5 + shift_scaled);
       diff = shift_scaled - int_shift_scaled; if (diff < 0) { diff = diff * -1; }
      if (diff < 0.1) { print 0.005 * int_shift_scaled; } else {
      print "get_frame_shift.sh: could not make sense of shift " shift_scaled  >"/dev/stderr"; exit(1) }}' || exit 1;

rm $temp

exit 0
