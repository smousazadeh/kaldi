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

nj=100

. ./path.sh
. ./cmd.sh
. utils/parse_options.sh


srcdir=exp/tri3

dir=exp/tri3_cleanup
lang=data/lang
data=data/train


if [ $stage -le 0 ]; then
  mkdir -p $dir
  cp $srcdir/final.mdl $dir
  cp $srcdir/tree $dir
  cp $srcdir/cmvn_opts $dir
  cp $srcdir/{splice_opts,final.mat,final.alimdl} $dir || true
fi

if [ $stage -le 1 ]; then
  echo "$0: Building biased-language-model decoding graphs..."
  steps/cleanup/make_biased_lm_graphs.sh \
    --nj $nj --cmd "$decode_cmd" \
    --top-n-words 100 \
    --min-words-per-graph 100 \
    --top-n-words-weight 1.0 \
     $data $lang $dir $dir
fi

if [ $stage -le 2 ]; then
  echo "$0: Decoding with biased-language-model decoding graphs..."
  steps/cleanup/decode_segmentation.sh \
    --nj $nj --cmd "$decode_cmd" \
    $dir $data $dir/lats
fi


if [ $stage -le 3 ]; then
  echo "$0: Doing oracle alignment of lattices..."
  steps/cleanup/lattice_oracle_align.sh --stage 5 \
    --cmd "$decode_cmd" $data $lang $dir/lats $dir/lattice_oracle
fi

if [ $stage -le 4 ]; then

  steps/cleanup/get_non_scored_words.py data/lang >

  steps/cleanup/modify_ctm_edits.py <() exp/tri3_cleanup/lattice_oracle/ctm_edits exp/tri3_cleanup/lattice_oracle/ctm_edits.modified.3b

# steps/cleanup/taint_ctm_edits.py exp/tri3_cleanup/lattice_oracle/ctm_edits.modified.3b exp/tri3_cleanup/lattice_oracle/ctm_edits.modified.4

  $decode_cmd $dir/log/do_resegmentation.py \
    steps/cleanup/modify_ctm_edits.py --lang=data/lang exp/tri3_cleanup/lattice_oracle/ctm_edits /dev/stdout \| \
    steps/cleanup/resegment_data.py /dev/stdin $dir/new_text $dir/sub_segments

  newdata=${data}_cleaned
  utils/copy_data_dir.sh $data $newdata


fi


exit 0

if [ $stage -le 4 ]; then
  steps/get_ctm.sh --cmd "$decode_cmd" --use-segments false \
    --print-silence true $data $lang $dir/lattice_oracle ${dataid}_appended $lang $output_dir
fi

data/train_si284_split \
    exp/tri2b/graph_train_si284_split exp/tri2b/decode_train_si284_split

exit 0;


if [ $stage -le 1 ]; then
  steps/cleanup/do_cleanup_segmentation.sh \
    --cmd "$train_cmd" --nj $nj \
    --stage $cleanup_stage \
    data/train data/lang $gmm_dir \
      ${gmm_dir}_cleanup
fi

if [ $stage -le 2 ]; then
  steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj \
    data/train_${cleanup_affix} exp/make_mfcc mfcc
  steps/compute_cmvn_stats.sh \
    data/train_${cleanup_affix} exp/make_mfcc mfcc
  utils/fix_data_dir.sh data/train_${cleanup_affix}
fi

nj=50

if [ $stage -le 3 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train_${cleanup_affix} data/lang exp/tri2 exp/tri2_ali_${cleanup_affix}
fi

if [ $stage -le 4 ]; then
  steps/train_sat.sh --cmd "$train_cmd" \
    5000 100000 data/train_${cleanup_affix} data/lang \
    exp/tri2_ali_${cleanup_affix} exp/tri3_${cleanup_affix}
fi

nj_dev=$(cat data/dev/spk2utt | wc -l)
nj_test=$(cat data/test/spk2utt | wc -l)

if [ $stage -le 5 ]; then
  graph_dir=exp/tri3_${cleanup_affix}/graph
  $decode_cmd $graph_dir/mkgraph.log \
    utils/mkgraph.sh data/lang_test exp/tri3_${cleanup_affix} $graph_dir

  steps/decode_fmllr.sh --nj $nj_dev --cmd "$decode_cmd" \
    $graph_dir data/dev exp/tri3_${cleanup_affix}/decode_dev
  steps/decode_fmllr.sh --nj $nj_test --cmd "$decode_cmd" \
    $graph_dir data/test exp/tri3_${cleanup_affix}/decode_test
fi
