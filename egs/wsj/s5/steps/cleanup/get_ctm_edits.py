#!/usr/bin/env python

# Copyright 2016   Vimal Manohar
#           2016   Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

from __future__ import print_function
import sys, operator, argparse

# Modify the CTM to include for each token the information from Levenshtein
# alignment of 'hypothesis' and 'reference'
# (i.e. the output of 'align-text' post-processed by 'wer_per_utt_details.pl')

# The information added to each token in the CTM is the reference word and one
# of the following edit-types:
#  'C' = correct
#  'S' = substitution
#  'D' = deletion
#  'I' = insertion
#  'SC' = silence and neighboring hypothesis words are both correct
#  'SS' = silence and one of the neighboring hypothesis words is a substitution
#  'SD' = silence and one of the neighboring hypothesis words is a deletion
#  'SI' = silence and one of the neighboring hypothesis words is an insertion
#  'XS' = reference word substituted by the OOV symbol in the hypothesis and both the neighboring hypothesis words are correct
# The priority order for the edit-types involving silence is 'SD' > 'SS' > 'SI'.

# See inline comments for details on how the CTM is processed.

# Note: Additional lines are added to the CTM to account for deletions.

# Input CTM:
# (note: the <eps> is for silence in the input CTM that comes from
# optional-silence in the graph.  However, the input edits don't have anything
# for these silences.

## TimBrown_2008P-0007226-0007620 1 0.000 0.100 when
## TimBrown_2008P-0007226-0007620 1 0.100 0.090 i
## TimBrown_2008P-0007226-0007620 1 0.190 0.300 some
## TimBrown_2008P-0007226-0007620 1 0.490 0.110 when
## TimBrown_2008P-0007226-0007620 1 0.600 0.060 i
## TimBrown_2008P-0007226-0007620 1 0.660 0.190 say
## TimBrown_2008P-0007226-0007620 1 0.850 0.450 go
## TimBrown_2008P-0007226-0007620 1 1.300 0.310 [COUGH]
## TimBrown_2008P-0007226-0007620 1 1.610 0.130 you
## TimBrown_2008P-0007226-0007620 1 1.740 0.180 got
## TimBrown_2008P-0007226-0007620 1 1.920 0.370 thirty
## TimBrown_2008P-0007226-0007620 1 2.290 0.830 seconds
## TimBrown_2008P-0007226-0007620 1 3.120 0.330 <eps>
## TimBrown_2008P-0007226-0007620 1 3.450 0.040 [BREATH]
## TimBrown_2008P-0007226-0007620 1 3.490 0.110 to
## TimBrown_2008P-0007226-0007620 1 3.600 0.320 [NOISE]

# Input Levenshtein edits : (the output of 'align-text' post-processed by 'wer_per_utt_details.pl')

## TimBrown_2008P-0007226-0007620 ref   ***  ***  [NOISE]  when  i  say  go  [COUGH]  you  [COUGH]   a    ve  got  thirty  seconds  [BREATH]  to  [NOISE]
## TimBrown_2008P-0007226-0007620 hyp  when   i     some   when  i  say  go  [COUGH]  you    ***    ***  ***  got  thirty  seconds  [BREATH]  to  [NOISE]
## TimBrown_2008P-0007226-0007620 op     I    I      S       C   C   C    C     C      C      D      D    D    C      C       C         C      C     C
## TimBrown_2008P-0007226-0007620 #csid 12 1 2 3

# Output.
# <file-id> <channel> <start-time> <end-time> <conf> <hyp-word> <ref-word> <edit>

## TimBrown_2008P-0007226-0007620 1 0.00 0.10 1.0 when *** I
## TimBrown_2008P-0007226-0007620 1 0.10 0.09 1.0 i *** I
## TimBrown_2008P-0007226-0007620 1 0.19 0.30 1.0 some [NOISE] S
## TimBrown_2008P-0007226-0007620 1 0.49 0.11 1.0 when when C
## TimBrown_2008P-0007226-0007620 1 0.60 0.06 1.0 i i C
## TimBrown_2008P-0007226-0007620 1 0.66 0.19 1.0 say say C
## TimBrown_2008P-0007226-0007620 1 0.84 0.45 1.0 go go C
## TimBrown_2008P-0007226-0007620 1 1.30 0.31 1.0 [COUGH] [COUGH] C
## TimBrown_2008P-0007226-0007620 1 1.61 0.13 1.0 you you C
## TimBrown_2008P-0007226-0007620 1 1.74 0.00 1.0 *** [COUGH] D
## TimBrown_2008P-0007226-0007620 1 1.74 0.00 1.0 *** a D
## TimBrown_2008P-0007226-0007620 1 1.74 0.00 1.0 <eps> ve D
## TimBrown_2008P-0007226-0007620 1 1.74 0.18 1.0 got got C
## TimBrown_2008P-0007226-0007620 1 1.92 0.37 1.0 thirty thirty C
## TimBrown_2008P-0007226-0007620 1 2.29 0.83 1.0 seconds seconds C
## TimBrown_2008P-0007226-0007620 1 3.12 0.33 1.0 <eps> <eps> SC
## TimBrown_2008P-0007226-0007620 1 3.45 0.04 1.0 [BREATH] [BREATH] C
## TimBrown_2008P-0007226-0007620 1 3.49 0.11 1.0 to to C
## TimBrown_2008P-0007226-0007620 1 3.60 0.32 1.0 [NOISE] [NOISE] C


def GetArgs():
    parser = argparse.ArgumentParser(description =
        """Append to the CTM the Levenshtein alignment of 'hypothesis' and 'reference' :
        (i.e. the output of 'align-text' post-processed by 'wer_per_utt_details.pl')""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--special-symbol", default = "***",
                        help = "Special symbol used to align insertion or deletion "
                        "in align-text binary")
    parser.add_argument("--silence-symbol", default = "<eps>",
                        help = "Must be provided to ignore silence words in the "
                        "CTM that would be present if --print-silence was true in "
                        "nbest-to-ctm binary")
    parser.add_argument("--oov-symbol", default = "<unk>",
                        help = "The OOV symbol; substitutions by oov are treated specially,"
                        "if you also supply the --symbol table option.")
    parser.add_argument("--symbol-table", type = str,
                        help = "The words.txt your system used; if supplied, it is used to "
                        "determine OOV words (and such words will count as correct if "
                        "substituted by the OOV symbol)")
    # Required arguments
    parser.add_argument("edits_in", metavar = "<edits-in>",
                        help = "Output of 'align-text' post-processed by 'wer_per_utt_details.pl'")
    parser.add_argument("ctm_in", metavar = "<ctm-in>",
                        help = "Hypothesized CTM")
    parser.add_argument("ctm_edits_out", metavar = "<ctm-eval-out>",
                        help = "CTM appended with word-edit information. ")
    args = parser.parse_args()

    return args

def CheckArgs(args):
    args.ctm_edits_out_handle = open(args.ctm_edits_out, 'w')

    if args.silence_symbol == args.special_symbol:
        sys.exit("get_ctm_edits.py: --silence-symbol and --special-symbol may not be the same",
                 file = sys.stderr)

    return args

kSilenceEdits = ['D', 'SC', 'SS', 'SD', 'SI']


# This map is used in PostProcessSilences to map from
# various contexts of silence to one of the special
# forms of silence.
kSilenceMap = { }


class CtmEditsProcessor:
    def __init__(self, args):
        self.silence_symbol = args.silence_symbol
        self.special_symbol = args.special_symbol
        self.oov_symbol = args.oov_symbol
        if args.symbol_table != None:
            LoadSymbolTable(args.symbol_table)

        # For each utterance-id utt, self.ctm[utt] will be an array of 6-tuples
        # (utterance-id, channel, begin-time-in-seconds, duration-in-seconds,
        # word, confidence), e.g ( 'utt1', '1', 16.2, 2.1, 'hello', 0.98 ).  The
        # begin and end times are floats but other fields are strings.  The
        # channel will usually be a dummy value like '1'; we keep it to be
        # compatible with the ctm format but it's not normally of interest.
        # We may not always have meaningful confidences; they will usually just be
        # 1.0.
        self.ctm = dict()
        # For each utterance-id utt, self.edits[utt] will
        # be an array of 3-tuples (op, hyp, ref), e.g.
        # ('I', 'hmm', '***'), or ('S' 'where', 'there').
        self.edits = dict()
        # For each utterance-id utt, self.ctm_edits[utt]
        # will be an array of 8-tuples (utterance-id, channel, begin-time-in-seconds,
        # duration-in-seconds, hyp-word, confidence, ref-word, operation ), e.g.
        # ( 'utt1', '1', 16.2, 2.1, 'there', 0.98, 'where', 'S' )
        self.ctm_edits = dict()      # key is the utt-id

    def LoadSymbolTable(symbol_table):
        self.word_list = set()
        try:
            f = open(symbol_table, 'r')
            for line in f.readlines():
                [ word, number ] = line.split()
                self.word_list.insert(word)
            assert len(self.word_list) > 0
        except:
            sys.exit("get_ctm_edits.py: error reading symbol table from " +
                     symbol_table)
        if not self.oov_symbol in self.word_list:
            sys.exit("get_ctm_edits.py: expected unknown-word symbol --oov-symbol={0} "
                     "to be in symbol table {1}".format(self.oov_symbol, self.symbol_table))



    def LoadEdits(self, eval_in):
        # Read the output of 'wer_per_utt_details.sh'.
        edits = self.edits
        with open(eval_in, 'r') as f:
            while True:
                # Reading 4 lines encoding one utterance,
                # e.g. ref = "utt-id1 ref  *** *** hello there you"
                ref = f.readline()
                # e.g. hyp = "utt-id1 ref  hmm hmm hello where ***"
                hyp = f.readline()
                # e.g. op = "utt-id1 ref   I   I    C    S  D"
                op = f.readline()
                # this line is not needed.
                csid = f.readline()
                if not ref: break
                # e.g. utt="utt-id1", tag = "ref", ref_str = "*** *** hello there you"
                utt,tag,ref_str = ref.split(' ', 2)
                assert(tag == 'ref')
                # e.g. utt="utt-id1", tag = "hyp", ref_str = "hmm hmm hello where ***"
                utt,tag,hyp_str = hyp.split(' ', 2)
                assert(tag == 'hyp')
                # e.g. utt="utt-id1", tag = "op", ref_str = "I I C S D"
                utt,tag,op_str = op.split(' ', 2)
                assert(tag == 'op')
                # e.g. ref_vec = [ '***', '***', 'hello', 'there', 'you' ]
                ref_vec = ref_str.split()
                # e.g. hyp_vec = [ 'hmm', 'hmm', 'hello', 'where', '**' ]
                hyp_vec = hyp_str.split()
                # e.g. op_vec = [ 'I', 'I', 'C', 'S', 'D' ]
                op_vec = op_str.split()
                assert(utt not in edits)
                # e.g. edits[utt] = [ ('I', 'hmm', '***'), ... ]
                edits[utt] = [ (op,hyp,ref) for op,hyp,ref in zip(op_vec, hyp_vec, ref_vec) ]

    def LoadCtm(self, ctm_in):
        # Load the 'ctm' into a dictionary,
        ctm = self.ctm
        with open(ctm_in) as f:
            for l in f:
                # e.g. splits = [ 'utt1', '1', '16.2', '2.1', 'hello', '0.98' ]
                splits = l.split()
                if len(splits) < 5 or len(splits > 6):
                    sys.exit("get_ctm_edits.py: bad line in input ctm file: " + l)
                if len(splits) == 5:
                    splits.append(1.0)  # let the confidence default to 1.
                utt, ch, beg, dur, wrd, conf = splits
                if not utt in ctm: ctm[utt] = []
                ctm[utt].append((utt, ch, float(beg), float(dur), wrd, float(conf)))

    # Process an insertion in the aligned text file. At an insertion, the ref
    # word is special_symbol. Processing steps are as follows:
    # 1) All the silences in the CTM just before this that were marked as 'SC'
    # will be changed to 'SI'
    # 2) All the silences in the CTM follwing this will be marked 'SI' with the
    # reference word as silence_symbol
    # 3) The line corresponding to the inserted hyp word, will be marked 'I'
    # with the reference word as silence_symbol
    # 4) The silences following the hyp word will be marked 'SI' with the
    # reference words as silence_symbol
    def ProcessInsertion(self, ctm, edits, ctm_index, edits_index, ctm_appended):
        assert(ctm_index < len(ctm))
        assert(edits_index < len(edits))
        assert (edits[edits_index][2] == self.special_symbol)  # ref word is special_symbol for Insertion

        i = len(ctm_appended) - 1
        while i >= 0 and ctm_appended[i][4] == self.silence_symbol:
            assert(ctm_appended[i][-1] in kSilenceEdits)
            if ctm_appended[i][-1] == 'SC':
                ctm_appended[i] = ctm_appended[i][0:-1] + ('SI',)
            i -= 1

        while (ctm[ctm_index][4] == self.silence_symbol):
            ctm_appended.append(ctm[ctm_index] +
                                (self.silence_symbol, 'SI'))
            ctm_index += 1
            assert(ctm_index < len(ctm))

        # hyp word must be in the CTM
        assert (ctm[ctm_index][4] == edits[edits_index][1])

        # Add silence_symbol as the reference word. This will probably not be
        # used anyway, so its ok to not use special_symbol
        ctm_appended.append(ctm[ctm_index] +
                (self.silence_symbol, 'I'))
        ctm_index += 1

        while ctm_index < len(ctm) and ctm[ctm_index][4] == self.silence_symbol:
            ctm_appended.append(ctm[ctm_index] +
                                (self.silence_symbol, 'SI'))
            ctm_index += 1

        edits_index += 1

        return ctm_index, edits_index

    # Process a substitution in the aligned text file.
    # Processing steps are as follows:
    # 1) All the silences in the CTM just before this that were marked as 'SC'
    # or 'SI' will be changed to 'SS'
    # 2) All the silences in the CTM follwing this will be marked 'SS' with the
    # reference word as silence_symbol
    # 3) The line corresponding to the substituted hyp word, will be marked 'S'
    # with the reference word as ref word
    # 4) The silences following the hyp word will be marked 'SS' with the
    # reference word as silence_symbol
    def ProcessSubstitution(self, ctm, edits, ctm_index, edits_index,
            ctm_appended):
        assert(ctm_index < len(ctm))
        assert(edits_index < len(edits))

        i = len(ctm_appended) - 1
        while i >= 0 and ctm_appended[i][4] == self.silence_symbol:
            assert(ctm_appended[i][-1] in kSilenceEdits)
            if ctm_appended[i][-1] in [ 'SI', 'SC' ]:
                ctm_appended[i] = ctm_appended[i][0:-1] + ('SS',)
            i -= 1

        while (ctm[ctm_index][4] == self.silence_symbol):
            ctm_appended.append(ctm[ctm_index] +
                                (self.silence_symbol, 'SS'))
            ctm_index += 1
            assert(ctm_index < len(ctm))

        # hyp word must be in the CTM
        assert(ctm[ctm_index][4] == edits[edits_index][1])

        if (ctm[ctm_index][4] == self.oov_symbol and
            ( (edits_index > 0 and edits[edits_index-1] == 'C')
                or (edits_index < len(edits)-1 and edits[edits_index+1] == 'C')
                or edits_index == 0 or edits_index == len(edits)-1 )
            ):
            # Substitution by an OOV is treated specially
            # if the adjacent words are both correct
            ctm_appended.append(ctm[ctm_index] +
                    (edits[edits_index][2], 'XS'))
        else:
            ctm_appended.append(ctm[ctm_index] +
                    (edits[edits_index][2], 'S'))
        ctm_index += 1

        while ctm_index < len(ctm) and ctm[ctm_index][4] == self.silence_symbol:
            ctm_appended.append(ctm[ctm_index] +
                                (self.silence_symbol, 'SS'))
            ctm_index += 1

        edits_index += 1
        return ctm_index, edits_index

    # Process a deletion in the aligned text file.
    # Processing steps are as follows:
    # 1) All the silences in the CTM before this will be processed only if it
    # has not been previously accounted for by some deletion. It will be
    # remarked as 'D' if there is no silence currently or we are at the end of
    # the utterance, in which case we must use the silence to account for the
    # current deletion. On the other hand, it will be remarked as 'SD' if there
    # is silence currently and we will be using the current silence to
    # account for the deletion in the subsequent steps.
    # 2) All the silences in the CTM follwing this will be marked 'D' with the
    # reference word as ref word if the deletion has not been previously
    # accounted for. A fake silence of 0s is added as necessary when there is
    # not silence around to put this deletion.
    # 3) The silences following the special_symbol hyp word will be marked 'SD'
    # although this might never happen.
    def ProcessDeletion(self, ctm, edits, ctm_index, edits_index, ctm_appended):
        assert(ctm_index <= len(ctm))
        assert(edits_index < len(edits))
        assert(edits[edits_index][1] == self.special_symbol)   # hyp word is special_symbol for Deletion

        i = len(ctm_appended) - 1
        deletion_accounted = False
        while i >= 0 and ctm_appended[i][4] == self.silence_symbol:
            assert(ctm_appended[i][-1] in kSilenceEdits)
            if ctm_appended[i][-1] != 'D':
                # If previous lines in ctm_appended do not correspond to a
                # previous deletion
                if (ctm_index == len(ctm) or
                        ctm[ctm_index][4] != self.silence_symbol):
                    # If we do not have another silence, we can just account for
                    # the deletion using the previous silence.
                    ctm_appended[i] = (ctm_appended[i][0:-2] +
                                        (edits[edits_index][2], 'D'))
                    deletion_accounted = True
                else:
                    # Here we are not yet accounting for the deletion. The
                    # previous SS or SC or SI is just remarked as SD.
                    ctm_appended[i] = (ctm_appended[i][0:-1] +
                                        ('SD',))
            i -= 1

        silence_conf = 0.01     # Not important for now as the confidence is not used for segmentation

        if not deletion_accounted:
            if ctm_index == len(ctm):
                # A deletion at the end of the utterance
                # Add a silence_symbol at the previous entry's end time with a
                # duration of 0s
                ctm_appended.append(ctm[ctm_index-1][0:2] +
                        (ctm[ctm_index-1][2]+ctm[ctm_index-1][3], 0,
                            self.silence_symbol, silence_conf,
                            edits[edits_index][2], 'D'))
            else:
                # If there is no silence in the CTM, associate the deletion with
                # a fake 0s silence created.
                ctm_appended.append(ctm[ctm_index][0:3] + (0, self.silence_symbol, silence_conf,
                    edits[edits_index][2], 'D'))
        elif (ctm_index < len(ctm) and ctm[ctm_index][4] == self.silence_symbol):
            assert(not deletion_accounted)
            # If there is a silence in the CTM, associate the deletion with
            # that silence
            ctm_appended.append(ctm[ctm_index] +
                    (edits[edits_index][2], 'D'))
            ctm_index += 1

        while ctm_index < len(ctm) and ctm[ctm_index][4] == self.silence_symbol:
            ctm_appended.append(ctm[ctm_index] +
                                (self.silence_symbol, 'SD'))
            ctm_index += 1

        edits_index += 1;
        return ctm_index, edits_index

    # Process a substitution in the aligned text file.
    # Appends to ctm_appended an 8-tuple which is formed of the
    # 6-tuple from the original ctm, plus the ref-word then the edit-type.
    # Processing steps are as follows:
    # 1) All the silences in the CTM just before this are left as is.
    # 2) All the silences in the CTM follwing this will be marked 'SC' with the
    # reference word as silence_symbol
    # 3) The line corresponding to the correct hyp word, will be marked 'C'
    # with the reference word as ref word
    # 4) The silences following the hyp word will be marked 'SC' with the
    # reference word as silence_symbol
    def ProcessCorrect(self, ctm, edits, ctm_index, edits_index, ctm_appended):
        assert(ctm_index < len(ctm))
        assert(edits_index < len(edits))

        while (ctm[ctm_index][4] == self.silence_symbol):
            ctm_appended.append(ctm[ctm_index] +
                                (self.silence_symbol, 'SC'))
            ctm_index += 1
            assert(ctm_index < len(ctm))

        assert(ctm[ctm_index][4] == edits[edits_index][1])
        assert(ctm[ctm_index][4] == edits[edits_index][2])

        ctm_appended.append(ctm[ctm_index] +
                (edits[edits_index][2], 'C'))
        ctm_index += 1

        while ctm_index < len(ctm) and ctm[ctm_index][4] == self.silence_symbol:
            ctm_appended.append(ctm[ctm_index] +
                                (self.silence_symbol, 'SC'))
            ctm_index += 1

        edits_index += 1
        return ctm_index, edits_index

    def ProcessSilence(self, ctm, edits, ctm_index, edits_index, ctm_appended):
        assert ctm[ctm_index][4] == self.silence_symbol
        # Pretend the reference word was the silence symbol (normally '<eps>')
        # and the silence was correct (hence 'SC').  Later we'll take errors into
        # account.
        ctm_appended.append(ctm[ctm_index] + (self.silence_symbol, 'SC'))

    def PostProcessSilences(self):
        ctm_edits = self.ctm_edits
        for utt in keys(ctm_edits):
            # first amalgamate successive silences so that two successive silences
            # never appear; this simplifies the later processing.
            utt_ctm_edits = ctm_edits[utt]
            new_ctm_edits = []
            for t in utt_ctm_edits:
                if t[7] == 'SC' and len(new_ctm_edits) > 0 and new_ctm_edits[-1][7] == 'SC':
                    # Amalgamate these two successive silences into one.
                    (utt1, chan1, start1, dur1, hyp1, conf1, ref1, op1) = t
                    (utt2, chan2, start2, dur2, hyp2, conf2, ref2, op2) = new_ctm_edits[-1]
                    assert hyp1 == hyp2 and hyp1 == self.silence_symbol
                    new_ctm_edits[-1] = (utt1, chan1, start1, dur1 + dur1, hyp1, 0.5 * (conf1 + conf2),
                                         ref1, op1)
                else:
                    new_ctm_edits.append(t)
            l = len(new_ctm_eits)
            for i in range(l)
                if new_ctm_edits[i][7] == 'SC':
                    prev_label = new_ctm_edits[i-1][7] if i > 0 else 'BOS'
                    next_label = new_ctm_edits[i+1][7] if i + 1 == l else 'EOS'
                    assert prev_label != 'SC' && next_label != 'SC'
                    index = prev_label + '+sil+' + next_label

            ctm_edits[utt] = new_ctm_edits

        # next,
        for utt in keys(ctm_edits):

    def PostProcessUnknownWords(self):
        if ! hasattr(self, word_list):
            # --symbol-table option was not provided.
            return
        for utt, utt_ctm_edits in ctm_edits.iteritems():
            for i in range(len(utt_ctm_edits)):
                (u, chan, start, dur, hyp, conf, ref, op) = utt_ctm_edits[i]
                if (hyp == self.oov_symbol and op == 'S' and
                    not ref in self.word_list):
                    # for OOV words that were decoded as unk, change the hyp to
                    # the reference and the label to 'C'.  Note: the lattice
                    # best-path will have favored these, because unkown words in
                    # the text will have been mapped to the OOV word.  We keep
                    # the hyp-word as <unk>, though, so it's easy to see what
                    # happened.
                    utt_ctm_edits[i] = (u, chan, start, dur, hyp, conf, ref, 'C')

    def CreateCtmEdits(self):
        # Build 'edits' which is the input ctm with an extra column for
        # 'edit-type' added; it may also have extra rows (i.e., extra lines,
        # once we print it out) corresponding to words that are in the reference
        # but not in the hypothesis.
        ctm = self.ctm
        edits = self.edits
        for utt, utt_ctm in ctm.iteritems():
            if not utt in edits:
                sys.exit("get_ctm_edits.py: utterance {0} does not appear in <edits-in> "
                         "input from {1}".format(utt, args.edits_in))
            utt_edits = edits[utt]
            # 'edits' is assumed to be in the correct order.
            ctm_index = 0
            edits_index = 0

            self.ctm_edits[utt] = []
            utt_ctm_edits = self.ctm_edits[utt]

            assert len(utt_ctm) != 0

            while edits_index < len(utt_edits) or ctm_index < len(utt_ctm):

                if ctm_index < len(utt_ctm) and utt_ctm[ctm_index][4] == self.silence_symbol:
                    # The optional silences won't be present in the output of
                    # 'align-text' so we just consume the ctm element.
                    utt_ctm_edits.append(ctm[ctm_index] + (self.silence_symbol, 'SC'))
                    ctm_index += 1
                    continue

                op,hyp,ref = utt_edits[edits_index]
                if op in [ 'I', 'S', 'C' ]:
                    # In all these cases, we consume a line from the ctm.  Check
                    # that the reference word in the edits is the same as the
                    # word in the input ctm.
                    assert ref == utt_edits[edits_index][2]
                    utt_ctm_edits.append(utt_ctm[ctm_index] + (ref, op))
                    ctm_index += 1
                    edits_index += 1
                else:
                    assert op == 'D'
                    adjacent_ctm_index = ctm_index if ctm_index < len(utt_ctm) else ctm_index - 1
                    (utt_id, channel, begin, duration, word, confidence) = utt_ctm[adjacent_ctm_index]
                    cur_begin_time = begin if adjacent_ctm_index == ctm_index else begin + duration
                    utt_ctm_edits.append( (utt_id, channel, cur_begin_time, 0.0, hyp, confidence, ref, 'D') )
                    edits_index += 1


    def WriteCtmEdits(self, ctm_edits_out_handle):
        for utt, utt_ctm_edits in self.ctm_edits.iteritems():
            for tup in sorted(utt_ctm_edits, key = lambda x:(x[2],x[2]+x[3])):
                try:
                    if len(tup) == 8:
                        ctm_edits_out_handle.write('%s %s %.02f %.02f %s %f %s %s\n' % tup)
                    else:
                        raise Exception("Invalid line in ctm-out {0}".format(str(tup)))
                except Exception:
                    raise Exception("Invalid line in ctm-out {0}".format(str(tup)))

def Main():
    print(" ".join(sys.argv), file = sys.stderr)

    args = GetArgs();
    args = CheckArgs(args)

    p = CtmEditsProcessor(args)
    p.LoadEdits(args.eval_in)
    p.LoadCtm(args.ctm_in)
    p.CreateCtmEdits()
    p.PostProcessSilences()
    p.WriteCtmEdits(args.ctm_edits_out_handle)

if __name__ == "__main__":
    Main()

