#!/bin/bash

# this script prepares the speed-perturbed mfcc training data if it doesn't
# already exist.

. ./cmd.sh
set -e

stage=1
train_stage=1
generate_alignments=true # false if doing ctc training
speed_perturb=true

feat_dim=40 # this is the MFCC dim we use in the hires features.  you can't change it
            # unless you change local/xvector/prepare_perturbed_data.sh to use a different
            # MFCC config with a different dimension.
data=data/train_nodup_sp_hires  # you can't change this without changing
                                # local/xvector/prepare_perturbed_data.sh
xvector_dim=200 # dimension of the xVector.  configurable.
xvector_dir=exp/xvector_a


. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

local/xvector/prepare_perturbed_data.sh --stage $stage


if [ $stage -le 3 ]; then
  # Prepare configs
  mkdir -p $xvector_dir/log

  $train_cmd $xvector_dir/log/make_configs.log \
    steps/nnet3/xvector/make_jesus_configs.py \
      --splice-indexes="-1,0,1 -2,-1,0,1 -3,0,3 mean+stddev+count(-99:3:9:0) 0" \
      --feat-dim $feat_dim --output-dim $xvector_dim \
      --num-jesus-blocks 100 \
      --jesus-input-dim 300 --jesus-output-dim 1000 --jesus-hidden-dim 2000 \
      $xvector_dir/nnet.config
fi

if [ $stage -le 4 ]; then
  # dump egs.
  steps/nnet3/xvector/get_egs.sh --cmd "$train_cmd" \
    "$data" $xvector_dir/egs
fi

exit 0;
