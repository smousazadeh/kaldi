#!/bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0

# This script demonstrates how to re-segment training data selecting only the
# "good" audio that matches the transcripts.
# The basic idea is to decode with an existing in-domain acoustic model, and a
# biased language model built from the reference, and then work out the
# segmentation from a ctm like file.

set -e -o pipefail

stage=0

pad_length=5                  # Number of frames for padding the created
                              # subsegments
max_silence_length=50         # Maxium number of silence frames above which they are removed and the segment is split
max_incorrect_words=2         # Maximum number of incorrect words allowed in the segments
min_correct_frames=0          # Minimum number of frames required in a subsegment to be kept
max_utterance_wer=20000       # Maximum WER% of an utterance, above which it is simply removed
min_wer_for_splitting=-1      # Minimum WER% for a segment to be considered for splitting

silence_padding_correct=5     # The amount of silence frames to pad a segment by
                              # if the silence is next to a correct hypothesis word
silence_padding_incorrect=20  # The amount of silence frames to pad a segment by
                              # if the silence is next to an incorrect hypothesis word

min_words_per_utt_group=200   # If provided, build LMs on group of utterances
                              # that have at least these many words.
                              # Otherwise, build per-utterance LM

# ngram options
ngram_order=2
top_n_words=100                       # Number of top-words to use for building a unigram LM
top_words_interpolation_weight=0.1    # Interpolation weight for top-words unigram
unigram_interpolation_weight=0.1      # Interpolation weight for unigram portion
                                      # of biased decoding graph that's
                                      # estimated from the data.

lmwt=10

cmd=run.pl
nj=4

. ./path.sh
. utils/parse_options.sh


if [ $# -ne 6 ]; then
  echo "Usage: $0 <data> <lang> <srcdir> <dir> <segmentation-dir> <out-data>"
  echo " e.g.: $0 data/sdm1/train_sp data/lang exp/sdm1/tri4a_ali_train_sp exp/sdm1/tri4a_sdm1_train_sp_bad_utts exp/sdm1/tri4a_sdm1/train_sp_bad_utts/segmentation_cleaned data/sdm1/train_sp_cleaned"
  exit 1
fi

data=$1
lang=$2
srcdir=$3
dir=$4
segmentation_dir=$5
out_data=$6

dataid=`basename $data`

mkdir -p $dir
cp $srcdir/final.mdl $dir
cp $srcdir/tree $dir
cp $srcdir/cmvn_opts $dir
cp $srcdir/{splice_opts,final.mat,final.alimdl} $dir || true


if [ $stage -le 2 ]; then
  echo "$0: Building biased-language-model decoding graphs..."
  steps/cleanup/make_biased_lm_graph.sh \
    --nj $nj --cmd "$cmd" \
    --ngram-order $ngram_order \
    --top-n-words $top_n_words \
    --min-words-per-utt-group $min_words_per_utt_group \
    --top-words-interpolation-weight $top_words_interpolation_weight \
    --biased-unigram-interpolation-weight $biased_unigram_interpolation_weight \
    --biased-bigram-interpolation-weight $biased_bigram_interpolation_weight \
    $dir/${dataid}_appended $lang $srcdir $dir/graphs_${dataid}
fi

if [ $stage -le 3 ]; then
  echo "$0: Decoding with biased language models..."
  steps/cleanup/decode_segmentation.sh \
    --nj $nj --cmd "$cmd --mem 4G" --skip-scoring true --allow-partial false \
    $dir/graphs_${dataid} \
    $dir/${dataid}_appended $dir/lats
fi

if [ $stage -le 4 ]; then
  echo "$0: Get oracle hypotheses from the decoded lattices..."
  steps/cleanup/lattice_oracle_align.sh \
    --cmd "$cmd" \
    --special-symbol "***" \
    $dir/${dataid}_appended $lang $dir
fi

output_dir=$dir/lattice_oracle

if [ $stage -le 5 ]; then
  # --use-segments false means we don't want the segmentation w.r.t. the
  # original recordings, but w.r.t. the orginal segments.
  steps/get_ctm.sh --cmd "$cmd" --use-segments false \
    --print-silence true \
    $dir/${dataid}_appended $lang $output_dir
fi

mkdir -p $segmentation_dir

if [ $stage -le 6 ]; then
  python steps/cleanup/get_ctm_edits.py \
    --special-symbol="***" --silence-symbol="<eps>" --oov-symbol="<unk>" \
    $output_dir/analysis/per_utt_details.txt \
    $output_dir/score_$lmwt/${dataid}_appended.ctm \
    $output_dir/score_$lmwt/${dataid}_appended.ctm.edits 2> $segmentation_dir/append_eval_to_ctm.log
fi

if [ $stage -le 7 ]; then
  echo "$0: Creating new segments in $segmentation_dir..."
  python steps/cleanup/create_segments_from_ctm_edits.py \
    --silence-symbol="<eps>" --oov-symbol="<unk>" \
    --pad-length=$pad_length \
    --max-silence-length=$max_silence_length \
    --min-wer-for-splitting=$min_wer_for_appendedting \
    --max-incorrect-words=$max_incorrect_words \
    --min-correct-frames=$min_correct_frames \
    --max-utterance-wer=$max_utterance_wer \
    --silence-padding-correct=$silence_padding_correct \
    --silence-padding-incorrect=$silence_padding_incorrect \
    $output_dir/score_$lmwt/${dataid}_appended.ctm.edits \
    $segmentation_dir/segments \
    $segmentation_dir/text 2>$segmentation_dir/segmentation.log
fi

if [ $stage -le 8 ]; then
  echo "$0: Resegmenting $dir/${dataid}_appended in $out_data..."
  steps/cleanup/resegment_data_dir.sh \
    $dir/${dataid}_appended $segmentation_dir/text \
    $segmentation_dir/segments ${out_data}
fi

echo "$0: Done cleaning up $data; new data dir created in $out_data"
