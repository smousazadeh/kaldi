#!/bin/bash

# Copyright 2016  Vimal Manohar
#           2016  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

# This script demonstrates how to re-segment training data selecting only the
# "good" audio that matches the transcripts.
# The basic idea is to decode with an existing in-domain acoustic model, and a
# biased language model built from the reference, and then work out the
# segmentation from a ctm like file.

set -e -o pipefail

stage=0

cmd=run.pl
cleanup=true
nj=4
acwt=0.08333 # you can set this e.g. to 0.15
beam=15.0
lattice_beam=6.0
graph_opts=

. ./path.sh
. utils/parse_options.sh


if [ $# -ne 4 ]; then
  echo "Usage: $0 [options] <data> <lang> <srcdir> <dir>"
  echo " This script finds likely bad utterances from data, for diagnostic"
  echo " purposes.  It's essentially the same as steps/cleanup/clean_and_segment_data.sh,"
  echo " but without actually doing anything, just getting diagnostics."
  echo ""
  echo "e.g. $0 data/train data/lang exp/tri3 exp/tri3_cleanup data/train_cleaned"
  echo "Options:"
  echo "  --stage <n>             # stage to run from, to enable resuming from partially"
  echo "                          # completed run (default: 0)"
  echo "  --cmd '$cmd'            # command to submit jobs with (e.g. run.pl, queue.pl)"
  echo "  --nj <n>                # number of parallel jobs to use in graph creation and"
  echo "                          # decoding"
  echo "  --graph-opts 'opts'     # Additional options to make_biased_lm_graphs.sh."
  echo "                          # Please run steps/cleanup/make_biased_lm_graphs.sh"
  echo "                          # without arguments to see allowed options."
  echo "  --cleanup <true|false>  # Clean up intermediate files afterward.  Default true."
  exit 1

fi

data=$1
lang=$2
srcdir=$3
dir=$4


for f in $srcdir/{final.mdl,tree,cmvn_opts} $data/utt2spk $data/feats.scp $lang/words.txt $lang/oov.txt; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist."
    exit 1
  fi
done

mkdir -p $dir
cp $srcdir/final.mdl $dir
cp $srcdir/tree $dir
cp $srcdir/cmvn_opts $dir
cp $srcdir/{splice_opts,delta_opts,final.mat,final.alimdl} $dir 2>/dev/null || true


if [ $stage -le 1 ]; then
  echo "$0: Building biased-language-model decoding graphs..."
  steps/cleanup/make_biased_lm_graphs.sh $graph_opts \
    --nj $nj --cmd "$decode_cmd" \
     $data $lang $dir
fi

if [ $stage -le 2 ]; then
  echo "$0: Decoding with biased language models..."
  transform_opt=
  if [ -f $srcdir/trans.1 ]; then
    # If srcdir contained trans.* then we assume they are fMLLR transforms for
    # this data, and we use them.
    transform_opt="--transform-dir $srcdir"
  fi
  # Note: the --beam 15.0 (vs. the default 13.0) does actually slow it
  # down substantially, around 0.35xRT to 0.7xRT on tedlium.
  # I want to test at some point whether it's actually necessary to have
  # this largish beam.
  steps/cleanup/decode_segmentation.sh --acwt "$acwt" \
      --beam "$beam" --lattice-beam "$lattice_beam" \
      --nj $nj --cmd "$cmd --mem 4G" $transform_opt \
      --skip-scoring true --allow-partial false \
       $dir $data $dir/lats

  # the following is for diagnostics, e.g. it will give us the lattice depth.
  steps/diagnostic/analyze_lats.sh --cmd "$cmd" $lang $dir/lats
fi

if [ $stage -le 3 ]; then
  echo "$0: Doing oracle alignment of lattices..."
  steps/cleanup/lattice_oracle_align.sh \
    --cmd "$decode_cmd" $data $lang $dir/lats $dir/lattice_oracle
fi

if [ $cleanup ]; then
  echo "$0: cleaning up intermediate files"
  rm -r $dir/fsts $dir/HCLG.fsts.scp
  rm -r $dir/lats/lat.*.gz $dir/lats/split_fsts
  rm $dir/lattice_oracle/lat.*.gz
fi

echo "$0: done."
