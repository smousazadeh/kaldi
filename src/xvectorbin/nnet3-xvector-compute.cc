// nnet3bin/nnet3-compute.cc

// Copyright 2012-2015   Johns Hopkins University (author: Daniel Povey)
//                2016   David Snyder

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
#include "base/timer.h"
#include "nnet3/nnet-utils.h"
#include "xvector/nnet-xvector-compute.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
      "Propagate the features through the network and write the output\n"
      "xvectors.  The xvector system assumes that for one input feature\n"
      "matrix, regardless of the number of rows, there is only one output\n"
      "xvector.  If the input has more rows than the network context, the\n"
      "extra rows will not be used in the xvector computation.\n"
      "\n"
      "Usage: nnet3-xvector-compute [options] <raw-nnet-in> "
      "<feats-rspecifier> <vector-wspecifier>\n"
      " e.g.: nnet3-xvector-compute final.raw scp:feats.scp "
      "ark:xvectors.ark\n";

    ParseOptions po(usage);
    Timer timer;

    NnetSimpleComputationOptions opts;
    std::string use_gpu = "yes";

    opts.Register(&po);

    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    std::string nnet_rxfilename = po.GetArg(1),
                feat_rspecifier = po.GetArg(2),
                vector_wspecifier = po.GetArg(3);

    Nnet nnet;
    ReadKaldiObject(nnet_rxfilename, &nnet);
    NnetXvectorComputer nnet_computer(opts, &nnet);

    BaseFloatVectorWriter vector_writer(vector_wspecifier);

    int32 num_success = 0, num_fail = 0;
    int64 frame_count = 0;

    SequentialBaseFloatMatrixReader feat_reader(feat_rspecifier);
    for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string utt = feat_reader.Key();
      const Matrix<BaseFloat> &feats (feat_reader.Value());
      if (feats.NumRows() == 0) {
        KALDI_WARN << "Zero-length utterance: " << utt;
        num_fail++;
        continue;
      }
      Vector<BaseFloat> xvector(nnet.OutputDim("output"));
      nnet_computer.ComputeXvector(feats, &xvector);
      vector_writer.Write(utt, xvector);
      frame_count += feats.NumRows();
      num_success++;
    }

    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken "<< elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed*100.0/frame_count);
    KALDI_LOG << "Done " << num_success << " utterances, failed for "
              << num_fail;

    if (num_success != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
