// xvectorbin/nnet3-xvector-merge-egs.cc

// Copyright      2016  David Snyder
//           2012-2015  Johns Hopkins University (author:  Daniel Povey)
//                2014  Vimal Manohar

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet3/nnet-example.h"
#include "nnet3/nnet-example-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;

    const char *usage =
        "This copies nnet examples for xvector training from input to\n"
        "output but while doing so it merges many NnetExample objects\n"
        "into one, forming a minibatch consisting of single NnetExample.\n"
        "Unlike nnet3-merge-egs, this binary does not expect the examples\n"
        "to have any output.\n"
        "\n"
        "Usage:  nnet3-xvector-merge-egs [options] <egs-rspecifier> "
        "<egs-wspecifier>\n"
        "e.g.\n"
        "nnet3-xvector-merge-egs --minibatch-size=512 ark:1.egs ark:- "
        "| nnet3-xvector-train ... \n"
        "See also nnet3-copy-egs and nnet3-merge-egs\n";

    bool compress = false;
    int32 minibatch_size = 512;
    bool discard_partial_minibatches = false;

    ParseOptions po(usage);
    po.Register("minibatch-size", &minibatch_size, "Target size of "
                "minibatches when merging.");
    po.Register("compress", &compress, "If true, compress the output examples "
                "(not recommended unless you are writing to disk)");
    po.Register("discard-partial-minibatches", &discard_partial_minibatches,
		"discard any partial minibatches of 'uneven' size that may be "
		"encountered at the end.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string examples_rspecifier = po.GetArg(1),
        examples_wspecifier = po.GetArg(2);

    SequentialNnetExampleReader example_reader(examples_rspecifier);
    NnetExampleWriter example_writer(examples_wspecifier);

    std::vector<NnetExample> examples;
    examples.reserve(minibatch_size);

    int32 num_read = 0, num_written = 0;
    while (!example_reader.Done()) {
      const NnetExample &cur_eg = example_reader.Value();
      examples.resize(examples.size() + 1);
      examples.back() = cur_eg;
      bool minibatch_ready =
          static_cast<int32>(examples.size()) >= minibatch_size;

      // Do Next() now, so we can test example_reader.Done() below .
      example_reader.Next();
      num_read++;

      if (minibatch_ready || (!discard_partial_minibatches &&
			      (example_reader.Done() && !examples.empty()))) {
        NnetExample merged_eg;
        MergeExamples(examples, compress, &merged_eg);
        std::ostringstream ostr;
        ostr << "merged-" << num_written;
        num_written++;
        std::string output_key = ostr.str();
        example_writer.Write(output_key, merged_eg);
        examples.clear();
      }
    }
    KALDI_LOG << "Merged " << num_read << " egs to " << num_written << '.';
    return (num_written != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


