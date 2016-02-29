// xvector/nnet-xvector-compute.cc

// Copyright      2015    Johns Hopkins University (author: Daniel Povey)
//                2016    David Snyder
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

#include "xvector/nnet-xvector-compute.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace nnet3 {

NnetXvectorComputer::NnetXvectorComputer(
    const NnetSimpleComputationOptions &config,
    Nnet *nnet):
    nnet_(nnet),
    config_(config),
    compiler_(*nnet, config.optimize_config) {
}

void NnetXvectorComputer::ComputeXvector(const MatrixBase<BaseFloat> &feats,
                    Vector<BaseFloat> *xvector) {

  ComputationRequest request;
  GetComputationRequest(feats, &request);
  const NnetComputation *computation = compiler_.Compile(request);
  NnetComputer computer(config_.compute_config, *computation,
                        *nnet_,
                        nnet_);
  std::string input_name = "input";
  CuMatrix<BaseFloat> cu_feats(feats);
  computer.AcceptInput(input_name, &cu_feats);
  computer.Forward();
  const CuMatrixBase<BaseFloat> &output = computer.GetOutput("output");
  KALDI_ASSERT(output.NumRows() == 1 && output.NumCols() == xvector->Dim());
  CuSubVector<BaseFloat> xvector_tmp(output, 0);
  xvector->CopyFromVec(xvector_tmp);
}

void NnetXvectorComputer::GetComputationRequest(
    const MatrixBase<BaseFloat> &feats,
    ComputationRequest *request) {
  std::string input_name = "input",
              output_name = "output";
  NnetIo nnet_io = NnetIo(input_name, 0, feats);
  request->inputs.clear();
  request->outputs.clear();
  request->inputs.resize(1);
  request->outputs.resize(1);
  request->need_model_derivative = false;
  request->store_component_stats = false;

  int32 input_node_index = nnet_->GetNodeIndex(input_name);

  if (input_node_index == -1 && !nnet_->IsInputNode(input_node_index))
    KALDI_ERR << "No input node called '" << input_name
              << "' in the network.";

  request->inputs[0].name = input_name;
  request->inputs[0].indexes = nnet_io.indexes;
  request->inputs[0].has_deriv = false;

  // We only need the output on frame t=0.
  int32 io_index_size = request->inputs[0].indexes.size(),
         n_indx_size = 1e6, t_ind;
  std::vector<Index> output_indexes;

  std::map<int32, int32> n_indx_sizes;
  for (int32 indx = 0; indx < io_index_size; indx++) {
    t_ind = request->inputs[0].indexes[indx].t;
    if (n_indx_sizes.count(t_ind) != 0)
      n_indx_sizes[t_ind] += 1;
    else
      n_indx_sizes.insert(std::make_pair(t_ind, 1));
  }
  std::map<int32, int32>::const_iterator iter;
  for (iter = n_indx_sizes.begin(); iter != n_indx_sizes.end(); iter++)
    n_indx_size = std::min(n_indx_size, iter->second);

  output_indexes.resize(n_indx_size);
  for (int32 indx = 0; indx < n_indx_size; indx++) {
    output_indexes[indx].n = indx;
    output_indexes[indx].t = 0;
  }

  // Add an io_spec for the output node.
  int32 output_node_index = nnet_->GetNodeIndex(output_name);
  if (!nnet_->IsOutputNode(output_node_index))
    KALDI_ERR << "No output node called '" << output_name
              << "' in the network.";
  request->outputs[0].name = output_name;
  request->outputs[0].indexes = output_indexes;
  request->outputs[0].has_deriv = false;

  // check to see if something went wrong.
  if (request->inputs.empty())
    KALDI_ERR << "No input in computation request.";
  if (request->outputs.empty())
    KALDI_ERR << "No output in computation request.";
}

} // namespace nnet3
} // namespace kaldi
