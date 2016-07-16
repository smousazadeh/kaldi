#!/bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0

# This script demonstrates how to re-segment training data selecting only the
# "good" audio that matches the transcripts.
# The basic idea is to decode with an existing in-domain acoustic model, and a
# biased language model built from the reference, and then work out the
# segmentation from a ctm like file.

# For nnet3 and chain results after cleanup, see the scripts in
# local/nnet3/run_tdnn.sh and local/chain/run_tdnn.sh


# GMM Results for speaker-independent (SI) and speaker adaptive training (SAT) systems on dev and test sets

# Baseline

# SI systems:

## %WER 65.6 | 14682 94521 | 41.2 40.5 18.3 6.8 65.6 73.0 | -22.334 | exp/sdm1/tri4a/decode_dev_ami_fsh.o3g.kn.pr1-7.si/ascore_12/dev_o4.ctm.filt.sys
## %WER 70.3 | 14505 90002 | 34.7 40.1 25.2 5.0 70.3 70.6 | -22.541 | exp/sdm1/tri4a/decode_eval_ami_fsh.o3g.kn.pr1-7.si/ascore_13/eval_o4.ctm.filt.sys

# SAT systems:

## %WER 64.1 | 14000 94540 | 43.0 39.2 17.9 7.1 64.1 76.5 | -22.276 | exp/sdm1/tri4a/decode_dev_ami_fsh.o3g.kn.pr1-7/ascore_12/dev_o4.ctm.filt.sys
## %WER 68.0 | 13862 89989 | 36.9 38.8 24.3 4.9 68.0 73.5 | -22.372 | exp/sdm1/tri4a/decode_eval_ami_fsh.o3g.kn.pr1-7/ascore_13/eval_o4.ctm.filt.sys

# Cleanup results
# local/run_cleanup_segmentation.sh --mic sdm1 --pad-length 5 --silence-padding-correct 5 --silence-padding-incorrect 20 --max-silence-length 100 --cleanup-affix cleaned_ah

# SI systems:

## %WER 64.8 | 13753 94528 | 41.6 39.3 19.1 6.4 64.8 78.1 | 0.513 | exp/sdm1/tri4a_cleaned_ah/decode_dev_ami_fsh.o3g.kn.pr1-7.si/ascore_12/dev_o4.ctm.filt.sys
## %WER 69.6 | 13576 89841 | 35.6 40.1 24.3 5.2 69.6 75.2 | 0.482 | exp/sdm1/tri4a_cleaned_ah/decode_eval_ami_fsh.o3g.kn.pr1-7.si/ascore_12/eval_o4.ctm.filt.sys

# SAT systems:

## %WER 62.6 | 13880 94530 | 43.5 37.4 19.1 6.1 62.6 77.6 | 0.414 | exp/sdm1/tri4a_cleaned_ah/decode_dev_ami_fsh.o3g.kn.pr1-7/ascore_12/dev_o4.ctm.filt.sys
## %WER 67.2 | 13545 89993 | 37.2 37.0 25.9 4.4 67.2 75.0 | 0.374 | exp/sdm1/tri4a_cleaned_ah/decode_eval_ami_fsh.o3g.kn.pr1-7/ascore_13/eval_o4.ctm.filt.sys


set -e
set -o pipefail
set -u

stage=0
cleanup_stage=0
mic=ihm

cleanup_affix=cleaned
nj=50
gmm=tri4a  # temp-- will make this tri4a.
data=data/$mic/train
lang=data/lang
srcdir=exp/$mic/$gmm
dir=exp/$mic/${gmm}_${cleanup_affix}
cleaned_data=${data}_${cleanup_affix}

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
    data/$mic/train_${cleanup_affix} data/lang exp/$mic/tri3a exp/$mic/tri3a_ali_${cleanup_affix}
fi

if [ $stage -le 3 ]; then
  steps/train_sat.sh --cmd "$train_cmd" \
    5000 80000 data/$mic/train_${cleanup_affix} data/lang exp/$mic/tri3a_ali_${cleanup_affix} exp/$mic/tri4a_${cleanup_affix}
fi

[ ! -r data/local/lm/final_lm ] && echo "Please, run 'run_prepare_shared.sh' first!" && exit 1
final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7

nj_dev=$(cat data/$mic/dev/spk2utt | wc -l)
nj_eval=$(cat data/$mic/eval/spk2utt | wc -l)

if [ $stage -le 4 ]; then
  graph_dir=exp/$mic/tri4a_${cleanup_affix}/graph_${LM}
  $decode_cmd $graph_dir/mkgraph.log \
    utils/mkgraph.sh data/lang_${LM} exp/$mic/tri4a_${cleanup_affix} $graph_dir
  steps/decode_fmllr.sh --nj $nj_dev --cmd "$decode_cmd" --config conf/decode.conf \
    $graph_dir data/$mic/dev exp/$mic/tri4a_${cleanup_affix}/decode_dev_${LM}
  steps/decode_fmllr.sh --nj $nj_eval --cmd "$decode_cmd" --config conf/decode.conf \
    $graph_dir data/$mic/eval exp/$mic/tri4a_${cleanup_affix}/decode_eval_${LM}
fi
