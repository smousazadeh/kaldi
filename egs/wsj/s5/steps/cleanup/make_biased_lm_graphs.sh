#!/bin/bash
# Copyright 2012-2016     Johns Hopkins University (Author: Daniel Povey)
#                2016     Vimal Manohar
# Apache 2.0


# This script creates biased decoding graphs based on the data transcripts as
# HCLG.fsts.scp, in the specified directory; this can be consumed by
# decode_segmentation.sh.
# This is for use in data-cleanup and data-filtering.


set -u
set -o pipefail
set -e

# Begin configuration section.
nj=10
cmd=run.pl
scale_opts="--transition-scale=1.0 --self-loop-scale=0.1"
top_n_words=100 # Number of common words that we compile into each graph (most frequent
                # in $data/text.orig.
top_n_words_weight=1.0
min_words_per_graph=100  # Utterances will be grouped so that they have at least
                         # this many words, before making the graph.
stage=-1
lm_opts=   # Additional options to make_biased_lm.py.

# End configuration options.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "usage: $0 <data-dir> <lang-dir> <src-dir> <dir>"
   echo "e.g.:  $0 data/train data/lang exp/tri1 exp/tri1/fsts"
   echo "  This script creates biased decoding graphs per utterance (or possibly"
   echo "  groups of utterances, depending on --min-words-per-graph).  Its output"
   echo "  goes to <dir>/HCLG.fsts.scp, indexed by utterance."
   echo "Main options (for others, see top of script file)"
   echo "  --scale-opts <scale-opts>                 # Options relating to language"
   echo "                                            # model scale; default is "
   echo "                                            # '--transition-scale=1.0 --self-loop-scale=0.1'"
   echo "  --top-n-words <N>                         # Number of most-common-words to add with"
   echo "                                            # unigram probabilities into graph (default: 100)"
   echo "  --top-n-words-weight <float>              # Weight given to top-n-words portion of graph"
   echo "                                            # (before renormalizing); may be any positive"
   echo "                                            # number (default: 1.0)"
   echo "  --lm-opts <opts>                          # Additional options to make_biased_lm.py"
   echo "  --config <config-file>                    # config containing options"
   echo "  --nj <nj>                                 # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
lang=$2
model_dir=$3
graph_dir=$4


for f in $lang/oov.int $model_dir/tree $model_dir/final.mdl \
    $lang/L_disambig.fst $lang/phones/disambig.int; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1;
done

oov=`cat $lang/oov.int` || exit 1;
mkdir -p $dir/log

# create top_words.{int,txt}
if [ $stage -le 0 ]; then
  export LC_ALL=C
  utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt <$data/text | \
    awk '{for(x=2;x<=NF;x++) print $x;}' | sort | uniq -c | \
    LC_ALL=C sort -nr | head -n $top_n_words > $dir/word_counts.int || exit 1;
  total_count=$(awk '{x+=$1} END{print x}' < $dir/word_counts.int) || exit 1;
  # print top-n words with their unigram probabilities.
  awk -v tot=$num_words -v weight=$top_n_words_weight '{print ($1*weight)/tot, $2;}' \
     <$dir/word_counts.int >$dir/top_words.int
  utils/int2sym.pl -f 2 $lang/words.txt <$dir/top_words.int >$dir/top_words.txt
fi

word_disambig_symbol=$(cat $lang/words.txt | grep -w "#0" | awk '{print $2}')
if [ -z "$word_disambig_symbol" ]; then
  echo "$0: error getting word disambiguation symbol"
  exit 1
fi

utils/split_data.sh --per-utt $data $nj
sdata=$data/split$nj  # caution: we'll have to change this when we
                      # change how --per-utt works.


mkdir -p $dir/log

if [ $stage -le 1 ]; then
  echo "$0: creating utterance-group-specific decoding graphs with biased LMs"

  $cmd JOB=1:$nj $dir/log/compile_decoding_graphs.JOB.log \
    utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt $sdata/JOB/text \| \
    python steps/cleanup/make_biased_lms.py --min-words-per-graph=$min_words_per_graph \
      --lm-opts="--word-disambig-symbol=$word_disambig_symbol $lm_opts" $dir/utt2group.JOB \| \
    compile-train-graphs-fsts $scale_opts --read-disambig-syms=$lang/phones/disambig.int \
      $model_dir/tree $model_dir/final.mdl $lang/L_disambig.fst ark:- \
    ark,scp:$dir/HCLG.fsts.JOB.ark,$dir/HCLG.fsts.JOB.scp || exit 1
fi

for j in $(seq $nj); do cat $dir/HCLG.fsts.$j.scp; done > $dir/HCLG.fsts.per_utt.scp
for j in $(seq $nj); do cat $dir/utt2group.$j; done > $dir/utt2group


# The following command
utils/apply_map.pl $dir/HCLG.fsts.per_utt.scp <$dir/utt2group > $dir/HCLG.fsts

n1=$(cat $data/utt2spk | wc -l)
n2=$(cat $dir/HCLG.fsts)

if [ $[$n1*9] -gt $[$n2*10] ]; then
  echo "$0: too many utterances have no scp, something seems to have gone wrong."
  exit 1
fi

exit 0;
