#!/bin/bash

# Copyright 2012-2015 Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.
#
# This script dumps training examples (egs) for xvector training.  These egs
# have only an input and no outputs (the inputs are typically MFCCs).  The egs
# consist of pairs of data chunks, with each chunk coming from different parts
# of the same utterance.  The two data-chunks in each eg will have respectively
# n=0 and n=1.  Each archive of egs has (in general) a different configuration,
# where a configuration is a pair of lengths of data-chunk (one for n=0 and one
# for n=1).  We don't mix together different lengths in the same archive,
# because it would require us to repeatedly run the compilation process within
# the same training job.
#
# This script, which will generally be called from other neural-net training
# scripts, extracts the training examples used to train the neural net (and also
# the validation examples used for diagnostics), and puts them in separate archives.
#
# This script dumps egs with several frames of labels, controlled by the
# frames_per_eg config variable (default: 8).  This takes many times less disk
# space because typically we have 4 to 7 frames of context on the left and
# right, and this ends up getting shared.  This is at the expense of slightly
# higher disk I/O while training.


# Begin configuration section.
cmd=run.pl
# each archive has two sets of data-chunks,
# each of length randomly chosen between $min_frames_per_eg and $max_frames_per_eg.
# (however the lengths do not differ within the archives, to avoid triggering
# excessive recompilation of computation graphs).
min_frames_per_chunk=50
max_frames_per_chunk=300
frames_per_iter=1000000 # have this many frames per archive.

frames_per_iter_diagnostic=100000 # have this many frames per archive for
                                  # the archives used for diagnostics.

num_diagnostic_archives=3  # we want to test the training and validation likelihoods
                           # on a range of utterance lengths, and this number controls
                           # how many archives we evaluate on.


compress=true   # set this to false to disable compression (e.g. if you want to see whether
                # results are affected).

num_heldout_utts=500     # number of utterances held out for validation.

num_repeats=1

stage=0
nj=6         # This should be set to the maximum number of jobs you are
             # comfortable to run in parallel; you can increase it if your disk
             # speed is greater and you have more machines.
stage=0

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 [opts] <data> <egs-dir>"
  echo " e.g.: $0 data/train exp/xvector_a/egs"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --nj <nj>                                        # The maximum number of jobs you want to run in"
  echo "                                                   # parallel (increase this only if you have good disk and"
  echo "                                                   # network speed).  default=6"
  echo "  --cmd (utils/run.pl;utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --min-frames-per-eg <#frames;50>                 # The minimum numer of frames per chunk that we dump"
  echo "  --max-frames-per-eg <#frames;200>                # The maximum numer of frames per chunk that we dump"
  echo "  --num-repeats <#repeats;1>                       # The (approximate) number of times the training"
  echo "                                                   # data is repeated in the egs"
  echo "  --frames-per-iter <#samples;1000000>             # Target number of frames per archive"
  echo "  --num-diagnostic-archives <#archives;3>          # Option that controls how many different versions of"
  echo "                                                   # the train and validation archives we create (e.g."
  echo "                                                   # train_subset.{1,2,3}.egs and valid.{1,2,3}.egs by default;"
  echo "                                                   # they contain different utterance lengths."
  echo "  --frames-per-iter-diagnostic <#samples;100000>   # Target number of frames for the diagnostic archives"
  echo "                                                   # {train_subset,valid}.*.egs"
  echo "  --stage <stage|0>                                # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."

  exit 1;
fi

data=$1
dir=$2

if [ ! -f $data/feats.scp ]; then
  echo "$0: expected $data/feats.scp to exist"
  exit 1
fi

if [ ! -f $data/utt2dur ]; then
  # getting this utt2dur will normally be more lightweight than
  # getting the exact utterance-to-length map.
  utils/data/get_utt2dur.sh $data/utt2dur || exit 1;
fi

frame_shift=$(utils/data/get_frame_shift.sh $data) || exit 1;
feat_dim=$(feat-to-len scp:$data/feat.scp) || exit 1

mkdir -p $dir/info $dir/log

echo $feat_dim > $dir/info/feat_dim

if [ $stage -le 0 ]; then
  echo "$0: getting utt2len file"
  # note: this utt2len file is only an approximation of the number of
  # frames in each file.
  cat $data/utt2dur | awk -v frame_shift=$frame_shift '{print $1, int($2 / frame_shift);}' > $dir/utt2len
fi

mkdir -p $dir/log $dir/info $dir/temp

temp=$dir/temp

# Get list of validation utterances.
awk '{print $1}' $data/utt2spk | utils/shuffle_list.pl | head -$num_utts_subset \
    > $temp/valid_uttlist || exit 1;

if [ -f $data/utt2uniq ]; then  # this matters if you use data augmentation.
  echo "File $data/utt2uniq exists, so augmenting valid_uttlist to"
  echo "include all perturbed versions of the same 'real' utterances."
  mv $temp/valid_uttlist $temp/valid_uttlist.tmp
  utils/utt2spk_to_spk2utt.pl $data/utt2uniq > $temp/uniq2utt
  cat $temp/valid_uttlist.tmp | utils/apply_map.pl $data/utt2uniq | \
    sort | uniq | utils/apply_map.pl $temp/uniq2utt | \
    awk '{for(n=1;n<=NF;n++) print $n;}' | sort  > $temp/valid_uttlist
  rm $temp/uniq2utt $temp/valid_uttlist.tmp
fi


utils/filter_scp.pl --exclude $temp/valid_uttlist <$temp/utt2len > $temp/utt2len.train
utils/filter_scp.pl $temp/valid_uttlist <$temp/utt2len > $temp/utt2len.valid


# first for the training data... work out how many archives.

num_train_frames=$(awk '{n += $2} END{print n}' <$temp/utt2len.train)
num_valid_frames=$(awk '{n += $2} END{print n}' <$temp/utt2len.valid)

echo $num_train_frames >$dir/info/num_frames

num_train_archives=$[($num_train_frames*$num_repeats)/$frames_per_iter + 1]
echo "$0: producing $num_train_archives archives for training"
echo $num_train_archives > $dir/info/num_archives





# the + 1 is to round up, not down... we assume it doesn't divide exactly.
num_archives=$[$num_frames/($frames_per_eg*$samples_per_iter)+1]
# (for small data)- while reduce_frames_per_eg == true and the number of
# archives is 1 and would still be 1 if we reduced frames_per_eg by 1, reduce it
# by 1.
reduced=false
while $reduce_frames_per_eg && [ $frames_per_eg -gt 1 ] && \
  [ $[$num_frames/(($frames_per_eg-1)*$samples_per_iter)] -eq 0 ]; do
  frames_per_eg=$[$frames_per_eg-1]
  num_archives=1
  reduced=true
done
$reduced && echo "$0: reduced frames_per_eg to $frames_per_eg because amount of data is small."

# We may have to first create a smaller number of larger archives, with number
# $num_archives_intermediate, if $num_archives is more than the maximum number
# of open filehandles that the system allows per process (ulimit -n).
max_open_filehandles=$(ulimit -n) || exit 1
num_archives_intermediate=$num_archives
archives_multiple=1
while [ $[$num_archives_intermediate+4] -gt $max_open_filehandles ]; do
  archives_multiple=$[$archives_multiple+1]
  num_archives_intermediate=$[$num_archives/$archives_multiple+1];
done
# now make sure num_archives is an exact multiple of archives_multiple.
num_archives=$[$archives_multiple*$num_archives_intermediate]

echo $num_archives >$dir/info/num_archives
echo $frames_per_eg >$dir/info/frames_per_eg
# Work out the number of egs per archive
egs_per_archive=$[$num_frames/($frames_per_eg*$num_archives)]
! [ $egs_per_archive -le $samples_per_iter ] && \
  echo "$0: script error: egs_per_archive=$egs_per_archive not <= samples_per_iter=$samples_per_iter" \
  && exit 1;

echo $egs_per_archive > $dir/info/egs_per_archive

echo "$0: creating $num_archives archives, each with $egs_per_archive egs, with"
echo "$0:   $frames_per_eg labels per example, and (left,right) context = ($left_context,$right_context)"



if [ -e $dir/storage ]; then
  # Make soft links to storage directories, if distributing this way..  See
  # utils/create_split_dir.pl.
  echo "$0: creating data links"
  utils/create_data_link.pl $(for x in $(seq $num_archives); do echo $dir/egs.$x.ark; done)
  for x in $(seq $num_archives_intermediate); do
    utils/create_data_link.pl $(for y in $(seq $nj); do echo $dir/egs_orig.$y.$x.ark; done)
  done
fi

if [ $stage -le 2 ]; then
  echo "$0: copying data alignments"
  for id in $(seq $num_ali_jobs); do gunzip -c $alidir/ali.$id.gz; done | \
    copy-int-vector ark:- ark,scp:$dir/ali.ark,$dir/ali.scp || exit 1;
fi

egs_opts="--left-context=$left_context --right-context=$right_context --compress=$compress"

[ -z $valid_left_context ] &&  valid_left_context=$left_context;
[ -z $valid_right_context ] &&  valid_right_context=$right_context;
valid_egs_opts="--left-context=$valid_left_context --right-context=$valid_right_context --compress=$compress"

echo $left_context > $dir/info/left_context
echo $right_context > $dir/info/right_context
num_pdfs=$(tree-info --print-args=false $alidir/tree | grep num-pdfs | awk '{print $2}')
if [ $stage -le 3 ]; then
  echo "$0: Getting validation and training subset examples."
  rm $dir/.error 2>/dev/null
  echo "$0: ... extracting validation and training-subset alignments."

  utils/filter_scp.pl <(cat $dir/valid_uttlist $dir/train_subset_uttlist) \
    <$dir/ali.scp >$dir/ali_special.scp

  $cmd $dir/log/create_valid_subset.log \
    nnet3-get-egs --num-pdfs=$num_pdfs $valid_ivector_opt $valid_egs_opts "$valid_feats" \
    "ark,s,cs:ali-to-pdf $alidir/final.mdl scp:$dir/ali_special.scp ark:- | ali-to-post ark:- ark:- |" \
    "ark:$dir/valid_all.egs" || touch $dir/.error &
  $cmd $dir/log/create_train_subset.log \
    nnet3-get-egs --num-pdfs=$num_pdfs $train_subset_ivector_opt $valid_egs_opts "$train_subset_feats" \
     "ark,s,cs:ali-to-pdf $alidir/final.mdl scp:$dir/ali_special.scp ark:- | ali-to-post ark:- ark:- |" \
     "ark:$dir/train_subset_all.egs" || touch $dir/.error &
  wait;
  [ -f $dir/.error ] && echo "Error detected while creating train/valid egs" && exit 1
  echo "... Getting subsets of validation examples for diagnostics and combination."
  $cmd $dir/log/create_valid_subset_combine.log \
    nnet3-subset-egs --n=$num_valid_frames_combine ark:$dir/valid_all.egs \
    ark:$dir/valid_combine.egs || touch $dir/.error &
  $cmd $dir/log/create_valid_subset_diagnostic.log \
    nnet3-subset-egs --n=$num_frames_diagnostic ark:$dir/valid_all.egs \
    ark:$dir/valid_diagnostic.egs || touch $dir/.error &

  $cmd $dir/log/create_train_subset_combine.log \
    nnet3-subset-egs --n=$num_train_frames_combine ark:$dir/train_subset_all.egs \
    ark:$dir/train_combine.egs || touch $dir/.error &
  $cmd $dir/log/create_train_subset_diagnostic.log \
    nnet3-subset-egs --n=$num_frames_diagnostic ark:$dir/train_subset_all.egs \
    ark:$dir/train_diagnostic.egs || touch $dir/.error &
  wait
  sleep 5  # wait for file system to sync.
  cat $dir/valid_combine.egs $dir/train_combine.egs > $dir/combine.egs

  for f in $dir/{combine,train_diagnostic,valid_diagnostic}.egs; do
    [ ! -s $f ] && echo "No examples in file $f" && exit 1;
  done
  rm $dir/valid_all.egs $dir/train_subset_all.egs $dir/{train,valid}_combine.egs
fi

if [ $stage -le 4 ]; then
  # create egs_orig.*.*.ark; the first index goes to $nj,
  # the second to $num_archives_intermediate.

  egs_list=
  for n in $(seq $num_archives_intermediate); do
    egs_list="$egs_list ark:$dir/egs_orig.JOB.$n.ark"
  done
  echo "$0: Generating training examples on disk"
  # The examples will go round-robin to egs_list.
  $cmd JOB=1:$nj $dir/log/get_egs.JOB.log \
    nnet3-get-egs --num-pdfs=$num_pdfs $ivector_opt $egs_opts --num-frames=$frames_per_eg "$feats" \
    "ark,s,cs:filter_scp.pl $sdata/JOB/utt2spk $dir/ali.scp | ali-to-pdf $alidir/final.mdl scp:- ark:- | ali-to-post ark:- ark:- |" ark:- \| \
    nnet3-copy-egs --random=true --srand=JOB ark:- $egs_list || exit 1;
fi

if [ $stage -le 5 ]; then
  echo "$0: recombining and shuffling order of archives on disk"
  # combine all the "egs_orig.*.JOB.scp" (over the $nj splits of the data) and
  # shuffle the order, writing to the egs.JOB.ark

  # the input is a concatenation over the input jobs.
  egs_list=
  for n in $(seq $nj); do
    egs_list="$egs_list $dir/egs_orig.$n.JOB.ark"
  done

  if [ $archives_multiple == 1 ]; then # normal case.
    $cmd --max-jobs-run $nj JOB=1:$num_archives_intermediate $dir/log/shuffle.JOB.log \
      nnet3-shuffle-egs --srand=JOB "ark:cat $egs_list|" ark:$dir/egs.JOB.ark  || exit 1;
  else
    # we need to shuffle the 'intermediate archives' and then split into the
    # final archives.  we create soft links to manage this splitting, because
    # otherwise managing the output names is quite difficult (and we don't want
    # to submit separate queue jobs for each intermediate archive, because then
    # the --max-jobs-run option is hard to enforce).
    output_archives="$(for y in $(seq $archives_multiple); do echo ark:$dir/egs.JOB.$y.ark; done)"
    for x in $(seq $num_archives_intermediate); do
      for y in $(seq $archives_multiple); do
        archive_index=$[($x-1)*$archives_multiple+$y]
        # egs.intermediate_archive.{1,2,...}.ark will point to egs.archive.ark
        ln -sf egs.$archive_index.ark $dir/egs.$x.$y.ark || exit 1
      done
    done
    $cmd --max-jobs-run $nj JOB=1:$num_archives_intermediate $dir/log/shuffle.JOB.log \
      nnet3-shuffle-egs --srand=JOB "ark:cat $egs_list|" ark:- \| \
      nnet3-copy-egs ark:- $output_archives || exit 1;
  fi

fi

if [ $stage -le 6 ]; then
  echo "$0: removing temporary archives"
  for x in $(seq $nj); do
    for y in $(seq $num_archives_intermediate); do
      file=$dir/egs_orig.$x.$y.ark
      [ -L $file ] && rm $(readlink -f $file)
      rm $file
    done
  done
  if [ $archives_multiple -gt 1 ]; then
    # there are some extra soft links that we should delete.
    for f in $dir/egs.*.*.ark; do rm $f; done
  fi
  echo "$0: removing temporary alignments and transforms"
  # Ignore errors below because trans.* might not exist.
  rm $dir/{ali,trans}.{ark,scp} 2>/dev/null
fi

echo "$0: Finished preparing training examples"
