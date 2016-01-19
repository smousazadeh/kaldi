// chain/chain-training.cc

// Copyright      2015   Johns Hopkins University (author: Daniel Povey)

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

#include "chain/chain-training.h"
#include "chain/chain-kernels-ansi.h"
#include "chain/chain-numerator.h"
#include "chain/chain-denominator.h"

namespace kaldi {
namespace chain {

void ComputeChainObjfAndDeriv(const ChainTrainingInfo &info,
                              const DenominatorGraph &den_graph,
                              const Supervision &supervision,
                              const CuMatrixBase<BaseFloat> &nnet_output,
                              BaseFloat *objf,
                              BaseFloat *l2_term,
                              BaseFloat *weight,
                              CuMatrixBase<BaseFloat> *nnet_output_deriv) {
  BaseFloat num_logprob_weighted;
  if (nnet_output_deriv)
    nnet_output_deriv->SetZero();
  {
    NumeratorComputation numerator(supervision, nnet_output);
    // note: supervision.weight is included as a factor in the derivative from
    // the numerator object, and the logprob too.
    num_logprob_weighted = numerator.Forward();
    if (nnet_output_deriv)
      numerator.Backward(nnet_output_deriv);
  }
  DenominatorComputation denominator(info, den_graph,
                                     supervision.num_sequences,
                                     nnet_output);

  BaseFloat den_logprob = denominator.Forward();
  bool ok = true;
  if (nnet_output_deriv)
    ok = denominator.Backward(-supervision.weight,
                              nnet_output_deriv);

  *objf = num_logprob_weighted - supervision.weight * den_logprob;
  *weight = supervision.weight * supervision.num_sequences *
      supervision.frames_per_sequence;
  if (!((*objf) - (*objf) == 0) || !ok) {
    // inf or NaN detected, or denominator computation returned false.
    if (nnet_output_deriv)
      nnet_output_deriv->SetZero();
    BaseFloat default_objf = -10;
    KALDI_WARN << "Objective function is " << (*objf)
               << " and denominator computation (if done) returned "
               << std::boolalpha << ok
               << ", setting objective function to " << default_objf
               << " per frame.";
    *objf  = default_objf * *weight;
  }

  // This code helps us see how big the derivatives are, on average,
  // for different frames of the sequences.  As expected, they are
  // smaller towards the edges of the sequences (due to the penalization
  // of 'incorrect' pdf-ids.
  if (GetVerboseLevel() >= 1) {
    int32 tot_frames = nnet_output_deriv->NumRows(),
 frames_per_sequence = supervision.frames_per_sequence,
       num_sequences = supervision.num_sequences;
    CuVector<BaseFloat> row_products(tot_frames);
    row_products.AddDiagMat2(1.0, *nnet_output_deriv, kNoTrans, 0.0);
    Vector<BaseFloat> row_products_cpu(row_products);
    Vector<BaseFloat> row_products_per_frame(frames_per_sequence);
    for (int32 i = 0; i < tot_frames; i++)
      row_products_per_frame(i / num_sequences) += row_products_cpu(i);
    KALDI_LOG << "Derivs per frame are " << row_products_per_frame;
  }

  ComputeL2Penalty(info, supervision.weight, nnet_output,
                   l2_term, nnet_output_deriv);
}


void ComputeL2Penalty(const ChainTrainingInfo &info,
                      BaseFloat supervision_weight,
                      const CuMatrixBase<BaseFloat> &nnet_output,
                      BaseFloat *l2_term,
                      CuMatrixBase<BaseFloat> *nnet_output_deriv) {
  if (info.l2_regularize == 0.0) {
    *l2_term = 0.0;
    return;
  }
  // compute the l2 penalty term and its derivative
  BaseFloat scale = supervision_weight * info.l2_regularize;
  *l2_term = -0.5 * scale * TraceMatMat(nnet_output, nnet_output, kTrans);
  if (nnet_output_deriv)
    nnet_output_deriv->AddMat(-1.0 * scale, nnet_output);
  if (info.two_level_tree_map.Dim() == 0)
    return;
  // The rest of this relates to the 2-level tree.
  int32 num_first_level_leaves = info.reverse_map.Dim();

  CuMatrix<BaseFloat> first_level_sums(nnet_output.NumRows(),
                                       num_first_level_leaves,
                                       kUndefined);
  // Set each column of 'first_level_sums' to the sum of the
  // columns in nnet_output that map to that leaf-index.
  first_level_sums.SumColumnRanges(nnet_output, info.reverse_map);

  // for each column col[i] of first_level_sums, we want to do:
  // *l2_term += 0.5 * scale * (n_i - 1)/n_i^2 * VecVec(col[i], col[i]),
  // where the factor of n_i - 1 is because we want to cancel out all but one of the
  // n_i such terms that we got from squared l2 norm of nnet_output, and the factor
  // of 1/n_i^2 is because we're dealing with sums not averages.
  // We deal with the factor of (n_i - 1)/n_i^2 by first scaling each
  // column by its square root.
  first_level_sums.MulColsVec(info.sqrt_scales);
  *l2_term += 0.5 * scale * TraceMatMat(first_level_sums, first_level_sums,
                                        kTrans);
  if (nnet_output_deriv) {
    // Apply the same scale again for the sake of the derivative computation.
    first_level_sums.MulColsVec(info.sqrt_scales);
    // Note the negation, we use 'scale' instead of '-1.0 * scale'.
    nnet_output_deriv->AddCols(scale, first_level_sums,
                               info.two_level_tree_map);

  }
}


// This function assumes forward_map is a vector containing zero-based indexes all >= 0 ,
// with the property that the same index only occurs in a contiguous range, e.g.
// forward_map = [  3 3 2 2 2 2 4 4 0 0 0 1 ].

// Each index should appear at least once.  Converts into a reversed map
// formatted as a set of [begin,end] pairs, e.g.  in this case
// [ (8, 11), (11, 12), (0, 2), (2, 6), (6, 8) ].
static void GetReverseMap(std::vector<int32> &forward_map,
                          std::vector<Int32Pair> *reverse_map) {
  int32 max_value = 0, size = forward_map.size();
  for (int32 i = 0; i < size; i++) {
    int32 n = forward_map[i];
    KALDI_ASSERT(n >= 0);
    if (n > max_value) max_value = n;
  }
  Int32Pair invalid_pair;
  invalid_pair.first = -1;
  invalid_pair.second = -1;
  reverse_map->resize(max_value + 1, invalid_pair);
  for (int32 i = 0; i < size; i++) {
    int32 n = forward_map[i];
    Int32Pair &p = (*reverse_map)[n];
    if (p.first == -1) {
      p.first = i;
      p.second = i + 1;
    } else {
      if (p.second != i)
        KALDI_ERR << "Tree map has non-contiguous ranges of the same value";
      p.second = i + 1;
    }
  }
  for (int32 i = 0; i <= max_value; i++) {
    // we don't actually require this property we're checking here (that all
    // level-1 tree indexes should appear, i.e. no gaps), but we do expect it.
    if ((*reverse_map)[i].first == -1)
      KALDI_ERR << "Value " << i << " does not appear in --two-level-tree-map.";
  }

}

ChainTrainingInfo::ChainTrainingInfo(
    const ChainTrainingOptions &info):
    l2_regularize(info.l2_regularize) {
  if (!info.two_level_tree_map_str.empty()) {
    bool binary;
    Input input(info.two_level_tree_map_str, &binary);
    std::vector<int32> two_level_tree_map_vec;
    // the next call will throw an error if there is a problem.
    ReadIntegerVector(input.Stream(), binary, &two_level_tree_map_vec);
    // this vector should not be empty if the option is supplied.
    KALDI_ASSERT(!two_level_tree_map_vec.empty());

    two_level_tree_map = two_level_tree_map_vec;

    // get reverse map.
    std::vector<Int32Pair> reverse_map_vec;
    GetReverseMap(two_level_tree_map_vec, &reverse_map_vec);
    reverse_map = reverse_map_vec;

    Vector<BaseFloat> sqrt_scales_vec(reverse_map_vec.size());
    for (int32 i = 0; i < sqrt_scales_vec.Dim(); i++) {
      BaseFloat n = reverse_map_vec[i].second - reverse_map_vec[i].first;
      KALDI_ASSERT(n >= 1.0);
      sqrt_scales_vec(i) = std::sqrt((n - 1) / (n * n));
    }
    sqrt_scales = sqrt_scales_vec;
  }
}


}  // namespace chain
}  // namespace kaldi
