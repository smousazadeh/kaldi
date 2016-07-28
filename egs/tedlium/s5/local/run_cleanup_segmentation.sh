#!/bin/bash

# Copyright 2016  Vimal Manohar
#           2016  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

# This script demonstrates how to re-segment training data selecting only the
# "good" audio that matches the transcripts.
# The basic idea is to decode with an existing in-domain acoustic model, and a
# biased language model built from the reference, and then work out the
# segmentation from a ctm like file.

# For nnet3 and chain results after cleanup, see the scripts in
# local/nnet3/run_tdnn.sh and local/chain/run_tdnn.sh

# GMM Results for speaker-independent (SI) and speaker adaptive training (SAT) systems on dev and test sets
# [will add these later].

set -e
set -o pipefail
set -u

stage=0
cleanup_stage=0
data=data/train
# note cleaned2 was with 1.0 min segment length,
# cleaned3 was with 0.5 min segment length,
# cleaned4 is with 0.5 generally but 1.0 for new segments. XXX due to bug it was 1.0 for *old* segments and 0.5 for new segments.
# cleaned5 is as cleaned4 but fix to that bug; is't now  0.5 generally but 1.0 for new segments.
cleanup_affix=cleaned5
cleaned_data=${data}_${cleanup_affix}
lang=data/lang
srcdir=exp/tri3
dir=exp/tri3_${cleanup_affix}_work

nj=100
decode_nj=8

. ./path.sh
. ./cmd.sh
. utils/parse_options.sh

if [ $stage -le 1 ]; then
  # This does the actual data cleanup.
  steps/cleanup/clean_and_segment_data.sh --stage $cleanup_stage --nj $nj --cmd "$train_cmd" \
    $data $lang $srcdir $dir $cleaned_data
fi

if [ $stage -le 2 ]; then
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    $cleaned_data $lang $srcdir ${srcdir}_ali_${cleanup_affix}
fi

if [ $stage -le 3 ]; then
  steps/train_sat.sh --cmd "$train_cmd" \
    5000 100000 $cleaned_data data/lang ${srcdir}_ali_${cleanup_affix} exp/tri4_${cleanup_affix}
fi

if [ $stage -le 4 ]; then
  # Test with the models trained on cleaned-up data.
  utils/mkgraph.sh data/lang_test exp/tri4_${cleanup_affix} exp/tri4_${cleanup_affix}/graph

  steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd" \
    --num-threads 8 \
    exp/tri4_${cleanup_affix}/graph data/dev exp/tri4_${cleanup_affix}/decode_dev
  steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd" \
    --num-threads 8 \
    exp/tri4_${cleanup_affix}/graph data/test exp/tri4_${cleanup_affix}/decode_test
fi

