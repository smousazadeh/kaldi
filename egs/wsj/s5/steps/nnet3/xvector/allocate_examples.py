#!/usr/bin/env python

# This script, for use when training xvectors, decides for you which examples
# will come from which utterances, and at what point.

# You call it as (e.g.)
#
#  allocate_examples.py --min-frames-per-chunk=50 --max-frames-per-chunk=200  --frames-per-iter=1000000 \
#   --num-archives=169 --num-jobs=24  exp/xvector_a/egs/temp/utt2len.train exp/xvector_a/egs/temp
#
#
# and this program outputs certain things to the temp directory (exp/xvector_a/egs/temp in this case)
# that will enable you to dump the xvectors.  What we'll eventually be doing is invoking the following
# program with something like the following args:
#
#  nnet3-xvector-get-egs1 [options] exp/xvector_a/temp/ranges.1  scp:data/train/feats.scp \
#    ark:exp/xvector_a/egs/egs_temp.1.ark ark:exp/xvector_a/egs/egs_temp.2.ark \
#    ark:exp/xvector_a/egs/egs_temp.3.ark
#
# where exp/xvector_a/temp/ranges.1 contains something like the following:
#
#   utt1  3   0   65  112  110
#   utt1  0   160 50  214  180
#   utt2  ...
#
# where each line is interpreted as follows:
#  <source-utterance> <output-archive-index>  <start-frame-index1> <num-frames1> <start-frame-index2> <num-frames2>
# and for each line we create an eg (containing two possibly-different-length chunks of data from the
# same utterance), to one of the output archives.  The list of archives corresponding to
# ranges.n will be written to output.n, so in exp/xvector_a/temp/outputs.1 we'd have:
#
#  ark:exp/xvector_a/egs/egs_temp.1.ark ark:exp/xvector_a/egs/egs_temp.2.ark ark:exp/xvector_a/egs/egs_temp.3.ark
#
# The number of these files will equal 'num-jobs'.  If you add up the word-counts of
# all the outputs.* files you'll get 'num-archives'.  The number of frames in each archive
# will be about the --frames-per-iter.
#
# This program will also output to the temp directory a file called archive_info which gives you
# the pairs of frame-lengths associated with each archives. e.g.
# 1   60  180
# 2   120  85
# the format is:  <archive-index> <num-frames1> <num-frames2>.
# the <num-frames1> and <num-frames2> will always be in the range
# [min-frames-per-chunk, max-frames-per-chunk].



# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings, random


parser = argparse.ArgumentParser(description="Writes ranges.*, outputs.* and archive_info files "
                                 "in preparation for dumping egs for xvector training."
                                 epilog="Called by steps/nnet3/xvector/get_egs.sh")
parser.add_argument("--min-frames-per-chunk", type=int, default=50,
                    help="Minimum number of frames-per-chunk used for any archive")
parser.add_argument("--max-frames-per-chunk", type=int, default=300,
                    help="Maximum number of frames-per-chunk used for any archive")
parser.add_argument("--frames-per-iter", type=int, default=1000000,
                    help="Target number of frames for each archive")
parser.add_argument("--num-archives", type=int, default=-1,
                    help="Number of archives to write");
parser.add_argument("--num-jobs", type=int, default=-1,
                    help="Number of jobs we're going to use to write the archives; the ranges.* "
                    "and outputs.* files are indexed by job.  Must be <= the --num-archives option.");
parser.add_argument("--seed", type=int, default=1,
                    help="Seed for random number generator")

# now the positional arguments
parser.add_argument("utt2len",
                    help="utt2len file of the features to be used as input (format is: "
                    "<utterance-id> <approx-num-frames>)");
parser.add_argument("output_dir",
                    help="Directory to write the output, ranges.*, outputs.* and archive_info");

print(' '.join(sys.argv))

args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

## Check arguments.
if args.min_frames_per_chunk <= 1:
    sys.exit("--min-frames-per-chunk is invalid.")
if args.max_frames_per_chunk < args.min_frames_per_chunk:
    sys.exit("--max-frames-per-chunk is invalid.")
if args.frames_per_iter < 1000:
    sys.exit("--frames-per-iter is invalid.")
if args.num_archives < 1:
    sys.exit("--num-archives is invalid")
if args.num_jobs > args.num_archives:
    sys.exit("--num-jobs is invalid (must not exceed num-archives)")


random.seed(args.seed)

f = open(args.utt2len, "r");
if f is None:
    sys.exit("Error opening utt2len file " + str(args.utt2len));

utt_ids = []
durations = []
for line in f:
    a = line.split()
    if len(a) != 2:
        sys.exit("bad line in utt2len file " + line);
    utt_ids.append(a[0])
    durations.append(a[1])

num_utts = len(utt_ids)
max_duration = max(durations)


if args.max_frames_per_chunk * 3 > max_duration:
    sys.exit("--max-frames-per-chunk={0} is not valid: it must be no more "
             "than a third of the maximum duration {1} from the utt2len file ".format(
            args.max_frames_per_chunk, max_duration))


# this function returns a random integer utterance index with probability
# proportional to its length.
def RandomUtt(min_length):
    while True:
        i = random.randrange(0, num_utts)
        # read the next line as 'with probability durations[i] / max_duration'.
        # this allows us to draw utterances with probability with
        # prob proportional to their length.
        if random.random() < durations[i] / max_duration:
            return i

# all_archives is an array of arrays, one per archive.
# Each archive's array is an array of 4-tuples
all_archives = []

for




# this is a bit like a struct, initialized from a string, which describes how to
# set up the statistics-pooling and statistics-extraction components.
# An example string is 'mean(-99:3:9::99)', which means, compute the mean of
# data within a window of -99 to +99, with distinct means computed every 9 frames
# (we round to get the appropriate one), and with the input extracted on multiples
# of 3 frames (so this will force the input to this layer to be evaluated
# every 3 frames).  Another example string is 'mean+stddev(-99:3:9:99)',
# which will also cause the standard deviation to be computed; or
# 'mean+stddev+count(-99:3:9:99)'.
class StatisticsConfig:
    # e.g. c = StatisticsConfig('mean+stddev(-99:3:9:99)', 400, 100, 'jesus1-output-affine')
    # e.g. c = StatisticsConfig('mean+stddev+count(-99:3:9:99)', 400, 100, 'jesus1-output-affine')
    def __init__(self, config_string, input_dim, num_jesus_blocks, input_name):
        self.input_dim = input_dim
        self.num_jesus_blocks = num_jesus_blocks  # we need to know this because
                                                  # it's the dimension of the count
                                                  # features that we output.
        self.input_name = input_name

        m = re.search("mean(|\+stddev)(|\+count)\((-?\d+):(-?\d+):(-?\d+):(-?\d+)\)",
                      config_string)
        if m == None:
            sys.exit("Invalid splice-index or statistics-config string: " + config_string)
        self.output_stddev = (m.group(1) == '+stddev')
        self.output_count = (m.group(1) == '+count')

        self.left_context = -int(m.group(2))
        self.input_period = int(m.group(3))
        self.stats_period = int(m.group(4))
        self.right_context = int(m.group(5))
        if not (self.left_context > 0 and self.right_context > 0 and
                self.input_period > 0 and self.stats_period > 0 and
                self.num_jesus_blocks > 0 and
                self.left_context % self.stats_period == 0 and
                self.right_context % self.stats_period == 0 and
                self.stats_period % self.input_period == 0):
            sys.exit("Invalid configuration of statistics-extraction: " + config_string)

    # OutputDim() returns the output dimension of the node that this produces.
    def OutputDim(self):
        return self.input_dim * (2 if self.output_stddev else 1) + (self.num_jesus_blocks if self.output_count else 0)


    # OutputDims() returns an array of output dimensions... this node produces
    # one output node, but this array explains how it's split up into different types
    # of output (which will affect how we reorder the indexes for the jesus-layer).
    def OutputDims(self):
        ans = [ self.input_dim ]
        if self.output_stddev:
            ans.append(self.input_dim)
        if self.output_count:
            ans.append(self.num_jesus_blocks)
        return ans

    # Descriptor() returns the textual form of the descriptor by which the
    # output of this node is to be accessed.
    def Descriptor(self):
        return 'Round({0}-pooling-{1}-{2}, {3})'.format(self.input_name, self.left_context,
                                                        self.right_context, self.stats_period)

    # This function writes the configuration lines need to compute the specified
    # statistics, to the file f.
    def WriteConfigs(self, f):
        print('component name={0}-extraction-{1}-{2} type=StatisticsExtractionComponent input-dim={3} '
              'input-period={4} output-period={5} include-variance={6} '.format(
                self.input_name, self.left_context, self.right_context,
                self.input_dim, self.input_period, self.stats_period,
                ('true' if self.output_stddev else 'false')), file=f)
        print('component-node name={0}-extraction-{1}-{2} component={0}-extraction-{1}-{2} input={0} '.format(
                self.input_name, self.left_context, self.right_context), file=f)
        stats_dim = 1 + self.input_dim * (2 if self.output_stddev else 1)
        print('component name={0}-pooling-{1}-{2} type=StatisticsPoolingComponent input-dim={3} '
              'input-period={4} left-context={1} right-context={2} num-log-count-features=0 '
              'output-stddevs={5} '.format(self.input_name, self.left_context, self.right_context,
                                           stats_dim, self.stats_period,
                                           ('true' if self.output_stddev else 'false')),
              file=f)
        print('component-node name={0}-pooling-{1}-{2} component={0}-pooling-{1}-{2} input={0}-extraction-{1}-{2} '.format(
                self.input_name, self.left_context, self.right_context), file=f)




## Work out splice_array
## e.g. for
## args.splice_indexes == '-3,-2,-1,0,1,2,3 -3,0:-3 -3,0:-3 -6,-3,0:-6,-3'
## we would have
##   splice_array = [ [ -3,-2,...3 ], [-3,0] [-3,0] [-6,-3,0]


splice_array = []
left_context = 0
right_context = 0
split_on_spaces = args.splice_indexes.split(" ");  # we already checked the string is nonempty.
if len(split_on_spaces) < 2:
    sys.exit("invalid --splice-indexes argument, too short: "
             + args.splice_indexes)
try:
    for string in split_on_spaces:
        this_layer = len(splice_array)

        this_splices = string.split(",")
        splice_array.append(this_splices)
        # the rest of this block just does some checking.
        for s in this_splices:
            try:
                n = int(s)
            except:
                if len(splice_array) == 1:
                    sys.exit("First dimension of splicing array must not have averaging [yet]")
                try:
                    x = StatisticsConfig(s, 100, 100, 'foo')
                except:
                    sys.exit("The following element of the splicing array is not a valid specifier "
                    "of statistics: " + s)

except ValueError as e:
    sys.exit("invalid --splice-indexes argument " + args.splice_indexes + " " + str(e))

num_hidden_layers = len(splice_array)


# all the remaining layers after the inputs in 'init.config' are added in one go.
f = open(args.config_dir + "/layers.config", "w")

print('input-node name=input dim=' + str(args.feat_dim), file=f)
cur_output = 'input'
cur_affine_output_dim = args.feat_dim

for l in range(1, num_hidden_layers + 1):
    # the following summarizes the structure of the layers:  Here, the Jesus component includes ReLU at its input and output, and renormalize
    #   at its output after the ReLU.
    # layer1: splice + affine + ReLU + renormalize
    # layerX: splice + Jesus + affine + ReLU

    # Inside the jesus component is:
    #  [permute +] ReLU + repeated-affine + ReLU + repeated-affine
    # [we make the repeated-affine the last one so we don't have to redo that in backprop].
    # We follow this with a post-jesus composite component containing the operations:
    #  [permute +] ReLU + renormalize
    # call this post-jesusN.
    # After this we use dim-range nodes to split up the output into
    # [ jesusN-output, jesusN-direct-output and jesusN-projected-output ]
    # parts;
    # and nodes for the jesusN-affine.

    print('# Config file for layer {0} of the network'.format(l), file=f)

    splices = []
    spliced_dims = []
    for s in splice_array[l-1]:
        # the connection from the previous layer
        try:
            offset = int(s)
            # it's an integer offset.
            splices.append('Offset({0}, {1})'.format(cur_output, offset))
            spliced_dims.append(cur_affine_output_dim)
        except:
            # it's not an integer offset, so assume it specifies the
            # statistics-extraction.
            stats = StatisticsConfig(s, cur_affine_output_dim,
                                     args.num_jeus_blocks, cur_output)
            stats.WriteConfigs(f)
            splices.append(stats.Descriptor())
            spliced_dims.extend(stats.OutputDims())

    # get the input to the Jesus layer.
    cur_input = 'Append({0})'.format(', '.join(splices))
    cur_dim = sum(spliced_dims)

    this_jesus_output_dim = args.jesus_output_dim

    # As input to the Jesus component we'll append the spliced input and any
    # mean/stddev-stats input, and the first thing inside the component that
    # we do is rearrange the dimensions so that things pertaining to a
    # particular block stay together.

    column_map = []
    for x in range(0, args.num_jesus_blocks):
        dim_offset = 0
        for src_splice in spliced_dims:
            src_block_size = src_splice / args.num_jesus_blocks
            for y in range(0, src_block_size):
                column_map.append(dim_offset + (x * src_block_size) + y)
            dim_offset += src_splice
    if sorted(column_map) != range(0, sum(spliced_dims)):
        print("column_map is " + str(column_map))
        print("num_jesus_blocks is " + str(args.num_jesus_blocks))
        print("spliced_dims is " + str(spliced_dims))
        sys.exit("code error creating new column order")

    need_input_permute_component = (column_map != range(0, sum(spliced_dims)))

    # Now add the jesus component.
    num_sub_components = (5 if need_input_permute_component else 4);
    print('component name=x-jesus{0} type=CompositeComponent num-components={1}'.format(
            l, num_sub_components), file=f, end='')
    # print the sub-components of the CompositeComopnent on the same line.
    # this CompositeComponent has the same effect as a sequence of
    # components, but saves memory.
    if need_input_permute_component:
        print(" component1='type=PermuteComponent column-map={1}'".format(
                l, ','.join([str(x) for x in column_map])), file=f, end='')
    print(" component{0}='type=RectifiedLinearComponent dim={1} self-repair-scale={2}'".format(
            (2 if need_input_permute_component else 1),
            cur_dim, args.self_repair_scale), file=f, end='')

    print(" component{0}='type=NaturalGradientRepeatedAffineComponent input-dim={1} output-dim={2} "
          "num-repeats={3} param-stddev={4} bias-mean={5} bias-stddev=0'".format(
            (3 if need_input_permute_component else 2),
            cur_dim, args.jesus_hidden_dim,
            args.num_jesus_blocks,
            args.jesus_stddev_scale / math.sqrt(cur_dim / args.num_jesus_blocks),
            0.5 * args.jesus_stddev_scale),
          file=f, end='')

    print(" component{0}='type=RectifiedLinearComponent dim={1} self-repair-scale={2}'".format(
            (4 if need_input_permute_component else 3),
            args.jesus_hidden_dim, args.self_repair_scale), file=f, end='')

    print(" component{0}='type=NaturalGradientRepeatedAffineComponent input-dim={1} output-dim={2} "
          "num-repeats={3} param-stddev={4} bias-mean={5} bias-stddev=0'".format(
            (5 if need_input_permute_component else 4),
            args.jesus_hidden_dim,
            this_jesus_output_dim,
            args.num_jesus_blocks,
            args.jesus_stddev_scale / math.sqrt(args.jesus_hidden_dim / args.num_jesus_blocks),
            0.5 * args.jesus_stddev_scale),
          file=f, end='')

    print("", file=f) # print newline.
    print('component-node name=x-jesus{0} component=x-jesus{0} input={1}'.format(
            l, cur_input), file=f)

    # now print the post-Jesus component which consists of ReLU +
    # renormalize.

    num_sub_components = 2
    print('component name=x-post-jesus{0} type=CompositeComponent num-components=2'.format(l),
          file=f, end='')

    # still within the post-Jesus component, print the ReLU
    print(" component1='type=RectifiedLinearComponent dim={0} self-repair-scale={1}'".format(
            this_jesus_output_dim, args.self_repair_scale), file=f, end='')
    # still within the post-Jesus component, print the NormalizeComponent
    print(" component2='type=NormalizeComponent dim={0} '".format(
            this_jesus_output_dim), file=f, end='')
    print("", file=f) # print newline.
    print('component-node name=x-post-jesus{0} component=x-post-jesus{0} input=x-jesus{0}'.format(l),
          file=f)

    cur_affine_output_dim = (args.jesus_input_dim if l < num_hidden_layers else args.final_hidden_dim)
    print('component name=x-affine{0} type=NaturalGradientAffineComponent '
          'input-dim={1} output-dim={2} bias-stddev=0'.
          format(l, args.jesus_output_dim, cur_affine_output_dim), file=f)
    print('component-node name=x-jesus{0}-output-affine component=x-affine{0} input=x-post-jesus{0}'.format(
            l), file=f)

    cur_output = 'x-jesus{0}-output-affine'.format(l)


print('component name=x-final-relu type=RectifiedLinearComponent dim={0} self-repair-scale={1}'.format(
        cur_affine_output_dim, args.self_repair_scale), file=f)
print('component-node name=x-final-relu component=x-final-relu input={0}'.format(cur_output),
      file=f)
print('component name=x-final-affine type=NaturalGradientAffineComponent '
      'input-dim={0} output-dim={1} bias-stddev=0'.format(
        cur_affine_output_dim, args.output_dim), file=f)
print('component-node name=x-final-affine component=x-final-affine input=x-final-relu',
      file=f)
print('component name=x-final-scale type=FixedScaleComponent dim={0} scale={1}'.format(
        args.output_dim, args.output_scale);
print('component-node name=x-final-scale component=x-final-scale input=x-final-affine',
      file=f)
print('output-node name=output input=x-final-scale', file=f)

# print components and nodes for the 'S' output (which is a vectorization of a
# symmetric-positive-definite matrix used in scoring), and the 'b' output (which
# is a scalar used in scoring).  Both of these are components of type
# ConstantFunctionComponent-- its output is a learned constant independent of
# the input-- so the input is pointless, but having an input keeps the component
# interface consistent and avoided us having to handle a graph with dangling
# nodes.

# First the S output...
s_dim = ((args.output_dim)+(args.output_dim+1))/2)
print('component name=x-s type=ConstantFunctionComponent input-dim={0} output-dim={1} '
      'output-mean=0 output-stddev=0 '.format(
            args.feat_dim, ((args.output_dim)+(args.output_dim+1))/2), file=f)
print('component-node name=x-s component=x-s input=IfDefined(input)',
      file=f)
print('component name=x-s-scale type=FixedScaleComponent dim={0} scale={1}'.format(
            s_dim, args.s_scale));
print('component-node name=x-s-scale component=x-s-scale input=x-s',
      file=f)
print('output-node name=s input=x-s-scale', file=f)

# now the 'b' output, which is just a scalar.
b_dim = 1
print('component name=x-b type=ConstantFunctionComponent input-dim={0} output-dim=1 '
      'output-mean=0 output-stddev=0 '.format(args.feat_dim), file=f)
print('component-node name=x-b component=x-b input=IfDefined(input)', file=f)
print('component name=x-b-scale type=FixedScaleComponent dim=1 scale={0}'.format(
        args.b_scale));
print('component-node name=x-b-scale component=x-b-scale input=input',
      file=f)
print('output-node name=b input=x-b-scale', file=f)
f.close()
