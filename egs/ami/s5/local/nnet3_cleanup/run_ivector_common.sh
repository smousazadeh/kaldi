#!/bin/bash

set -e -o pipefail


# this is a simplified version of local/nnet3/run_ivector_common.sh (the part
# that deals with data preparation), that does not support using parallel
# alignments and allows you to specify the source GMM-dir directly.  It's mainly
# intended for use with the data-cleanup script.


# e.g.:
# local/nnet3_cleanup/run_ivector_common.sh --mic ihm --affix _cleaned2 --gmm tri5a_cleaned2 --train-set train_cleaned2


stage=0
mic=ihm
nj=30
min_seg_len=1.55
train_set=train # you might set this to e.g. train_cleaned.
gmm=tri4a  # this might become e.g. tri5a_cleaned.

num_threads_ubm=32
affix=   # affix for exp/$mic/nnet3 directory to put iVector stuff in, so it
         # becomes exp/$mic/nnet3_cleaned or whatever.

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh



gmmdir=exp/${mic}/${gmm}
alidir=${gmmdir}_ali_sp_comb

for f in data/${mic}/${train_set}/feats.scp ${gmmdir}/final.mdl; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done


if [ $stage -le 1 ]; then
  echo "$0: preparing speed-perturbed data"
  utils/data/perturb_data_dir_speed_3way.sh data/${mic}/${train_set} data/${mic}/${train_set}_sp
fi

if [ $stage -le 2 ]; then
  if [ -f data/${mic}/${train_set}_sp/feats.scp ]; then
    echo "$0: $data_perturbed/feats.scp already exists.  Refusing to overwrite them "
    echo " to avoid wasting time.  Please remove the file and continue if you really mean this."
    exit 1;
  fi
  echo "$0: making MFCC features for speed-perturbed data"
  steps/make_mfcc.sh --nj $nj \
    --cmd "$train_cmd" data/${mic}/${train_set}_sp
  steps/compute_cmvn_stats.sh data/${mic}/${train_set}_sp

  echo "$0: fixing input data-dir in case some segments were too short."
  utils/fix_data_dir.sh data/${mic}/${train_set}_sp
fi

if [ $stage -le 3 ]; then
  echo "$0: combining short segments of speed-perturbed data"
  utils/data/combine_short_segments.sh \
      data/${mic}/${train_set}_sp $min_seg_len data/${mic}/${train_set}_sp_comb
  # re-use the CMVN stats from the source directory, since it seems to be slow to
  # re-compute them after concatenating short segments.
  cp data/${mic}/${train_set}_sp/cmvn.scp data/${mic}/${train_set}_sp_comb/
  utils/fix_data_dir.sh data/${mic}/${train_set}_sp_comb/
fi


if [ $stage -le 4 ]; then
  echo "$0: aligning perturbed, short-segment-combined data"
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
     data/$mic/${train_set}_sp_comb data/lang $gmmdir $alidir
fi

if [ $stage -le 5 ]; then
  echo "$0: creating high-resolution MFCC features"
  # this shows how you can split across multiple file-systems.  we'll split the
  # MFCC dir across multiple locations.  You might want to be careful here, if you
  # have multiple copies of Kaldi checked out and run the same recipe, not to let
  # them overwrite each other.
  mfccdir=data/$mic/${train_set}_sp_hires/data
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/egs/ami-$mic-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage $mfccdir/storage
  fi

  for datadir in ${train_set}_sp dev eval; do
    utils/copy_data_dir.sh data/$mic/$datadir data/$mic/${datadir}_hires
  done

  # do volume-perturbation on the training data prior to extracting hires
  # features; this helps make trained nnets more invariant to test data volume.
  utils/data/perturb_data_dir_volume.sh data/$mic/${train_set}_sp_hires

  for datadir in ${train_set}_sp dev eval; do
    steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/$mic/${datadir}_hires
    steps/compute_cmvn_stats.sh data/$mic/${datadir}_hires
    utils/fix_data_dir.sh data/$mic/${datadir}_hires
  done

  # call utils_fix_data_dir.sh on the hires data, because after speed perturbation, some
  # segments go below the minimum segment length
  utils/fix_data_dir.sh data/${mic}/${train_set}_sp_hires
fi

if [ $stage -le 6 ]; then
  echo "$0: combining short segments of speed-perturbed high-resolution MFCC training data"
  # we have to combine short segments or we won't be able to train chain models
  # on those segments.
  utils/data/combine_short_segments.sh \
     data/${mic}/${train_set}_sp_hires $min_seg_len data/${mic}/${train_set}_sp_hires_comb

  # just copy over the CMVN to avoid having to recompute it.
  cp data/${mic}/${train_set}_sp_hires/cmvn.scp data/${mic}/${train_set}_sp_hires_comb/
  utils/fix_data_dir.sh data/${mic}/${train_set}_sp_hires_comb/

fi


if [ $stage -le 7 ]; then
  # Train a system on top of a subset of the perturbed hires data just for its
  # LDA+MLLT transform.  We use --num-iters 13 because after we get the
  # transform (12th iter is the last), any further training is pointless.
  # We don't need to use the segment-combined data for this.

  # select about a half of the data.
  num_utts_total=$(wc -l <data/$mic/${train_set}_sp_hires/utt2spk)
  num_utt=$[$num_utts_total/2]

  utils/data/subset_data_dir.sh data/$mic/${train_set}_sp_hires $num_utt data/$mic/${train_set}_sp_hires_subset


  if [ -d exp/$mic/nnet3${affix}/tri5/final.mdl ]; then
    # we don't want to overwrite old stuff, ask the user to delete it.
    echo "$0: exp/$mic/nnet3${affix}/tri5/final.mdl already exists: please delete and then rerun."
    exit 1;
  fi

  steps/train_lda_mllt.sh --cmd "$train_cmd" --num-iters 13 \
      --realign-iters "" \
      --splice-opts "--left-context=3 --right-context=3" \
      5000 10000 data/$mic/${train_set}_sp_hires_subset data/lang \
      $alidir exp/$mic/nnet3${affix}/tri5
fi


if [ $stage -le 8 ]; then
  # train a diagonal UBM, again using the subset.
  # Use 512 Gaussians in the UBM.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 \
    --num-frames 700000 \
    --num-threads $num_threads_ubm \
    data/$mic/${train_set}_sp_hires_subset 512 exp/$mic/nnet3${affix}/tri5 exp/$mic/nnet3${affix}/diag_ubm
fi

if [ $stage -le 9 ]; then
  # Train the iVector extractor.  Use all of the speed-perturbed data since iVector extractors
  # can be sensitive to the amount of data.  The script defaults to an iVector dimension of
  # 100.
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
    data/$mic/${train_set}_sp_hires exp/$mic/nnet3${affix}/diag_ubm exp/$mic/nnet3${affix}/extractor || exit 1;
fi

if [ $stage -le 10 ]; then
  # note, we don't encode the 'max2' in the ivectordir even though that's the data
  # we extract the ivectors from, as it's still going to be valid for the non-'max2' data,
  # the utterance list is the same.
  ivectordir=exp/$mic/nnet3${affix}/ivectors_${train_set}_sp_hires_comb
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $ivectordir/storage ]; then
    utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/egs/ami-$mic-$(date +'%m_%d_%H_%M')/s5/$ivectordir/storage $ivectordir/storage
  fi
  # We extract iVectors on the speed-perturbed training data after combining
  # short segments, which will be what we train the system on.  With
  # --utts-per-spk-max 2, the script pairs the utterances into twos, and treats
  # each of these pairs as one speaker; this gives more diversity in iVectors..
  # Note that these are extracted 'online'.

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 \
       data/${mic}/${train_set}_sp_hires_comb data/${mic}/${train_set}_sp_hires_comb_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
    data/${mic}/${train_set}_sp_hires_comb_max2 \
    exp/$mic/nnet3${affix}/extractor $ivectordir

  # do the same for the test data, but in this case we don't need the speed
  # perturbation (sp) or small-segment concatenation (comb).
  for data in dev eval; do
    nj=$(wc -l < data/${mic}/${data}_hires/spk2utt)
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj "$nj" \
      data/${mic}/${data}_hires exp/$mic/nnet3${affix}/extractor \
      exp/$mic/nnet3${affix}/ivectors_${data}_hires
  done
fi

exit 0;
