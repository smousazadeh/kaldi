// chain/chain-training.h

// Copyright       2015  Johns Hopkins University (Author: Daniel Povey)


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


#ifndef KALDI_CHAIN_CHAIN_TRAINING_H_
#define KALDI_CHAIN_CHAIN_TRAINING_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "tree/context-dep.h"
#include "lat/kaldi-lattice.h"
#include "matrix/kaldi-matrix.h"
#include "hmm/transition-model.h"
#include "chain/chain-den-graph.h"
#include "chain/chain-supervision.h"

namespace kaldi {
namespace chain {


struct ChainTrainingOptions {
  // Currently empty.  l2 regularization constant; the actual term added to the
  // objf will be -0.5 times this constant times the squared l2 norm.  (squared
  // so it's additive across the dimensions).
  BaseFloat l2_regularize;
  std::string two_level_tree_map_str;
  BaseFloat two_level_tree_scale;

  ChainTrainingOptions(): l2_regularize(0.0), two_level_tree_scale(1.0) { }

  void Register(OptionsItf *opts) {
    opts->Register("l2-regularize", &l2_regularize, "l2 regularization "
                   "constant for 'chain' training, applied to the output "
                   "of the neural net.");
    opts->Register("two-level-tree-map", &two_level_tree_map_str,
                   "Filename for map from second-level to first tree if "
                   "using two-level tree (affects how l2 regularization is "
                   "applied)");
    opts->Register("two-level-tree-scale", &two_level_tree_scale,
                   "Scaling factor on the two-level-tree-map modification, where "
                   "1 gives you full application and 0 gives non at all.");
  }
};


/**
   This class represents a processed form of the command-line options from
   ChainTrainingOptions-- it contains some things we need to precompute,
   relating to the l2 regularization with two-level trees.
*/
struct ChainTrainingInfo {
  BaseFloat l2_regularize;
  BaseFloat two_level_tree_scale;

  // If the two-level-tree-map option is not given, the arrays below will
  // all be empty.

  // If --two-level-tree-map option is given, maps from 2nd-level to 1st-level
  // tree (else empty).  Dimension is number of pdfs in 2nd-level tree (i.e. the
  // tree the model was built with).
  CuArray<int32> two_level_tree_map;

  // If the --two-level-tree-map option is given, this contains a reverse map
  // from 1st-level tree index to (start, end) of ranges of 2nd-level tree
  // indexes. (the two-level tree building code ensures that the indexes form
  // ranges like this).  Needed in derivative computation.
  CuArray<Int32Pair> reverse_map;


  // If the --two-level-tree-map option is given, this vector contains a
  // quantity that's used in the l2 regularization computation.  The dimension
  // of this vector is the same as num-leaves in the 1st-level tree, and
  // sqrt_scales(i) = sqrt( (n_i-1) / n_i^2 ), where n_i is the number of 2nd-level
  // tree leaves that map to leaf i of the first-level tree.
  CuVector<BaseFloat> sqrt_scales;


  explicit ChainTrainingInfo(const ChainTrainingOptions &opts);

};


/**
   This function does both the numerator and denominator parts of the 'chain'
   computation in one call.

   @param [in] opts        Struct containing options
   @param [in] den_graph   The denominator graph, derived from denominator fst.
   @param [in] supervision  The supervision object, containing the supervision
                            paths and constraints on the alignment as an FST
   @param [in] nnet_output  The output of the neural net; dimension must equal
                          ((supervision.num_sequences * supervision.frames_per_sequence) by
                            den_graph.NumPdfs()).  The rows are ordered as: all sequences
                            for frame 0; all sequences for frame 1; etc.
   @param [out] objf       The [num - den] objective function computed for this
                           example; you'll want to divide it by 'tot_weight' before
                           displaying it.
   @param [out] l2_term  The l2 regularization term in the objective function, if
                           the --l2-regularize option is used.  To be added to 'o
   @param [out] weight     The weight to normalize the objective function by;
                           equals supervision.weight * supervision.num_sequences *
                           supervision.frames_per_sequence.
   @param [out] nnet_output_deriv  The derivative of the objective function w.r.t.
                           the neural-net output.  Only written to if non-NULL.
                           You don't have to zero this before passing to this function,
                           we zero it internally.
*/
void ComputeChainObjfAndDeriv(const ChainTrainingInfo &info,
                              const DenominatorGraph &den_graph,
                              const Supervision &supervision,
                              const CuMatrixBase<BaseFloat> &nnet_output,
                              BaseFloat *objf,
                              BaseFloat *l2_term,
                              BaseFloat *weight,
                              CuMatrixBase<BaseFloat> *nnet_output_deriv);

/**
   You won't need to call this function yourself (it is called from
   ComputeChainObjfAndDeriv), but we break it out in order to document the
   l2-regularization behavior.

   This function computes the l2 regularization term, sets *l2_term to that
   value, and if nnet_output_deriv != NULL, adds the derivative w.r.t. that
   term to *nnet_output_deriv.  If the --two-level-tree-map option was
   not supplied, the l2 regularization term equals
       -0.5 * info.l2_regularize * supervision_weight *
           TraceMatMat(nnet_output, nnet_output, kTarns).
   where the TraceMatMat term gives you the squared l2 norm of the output.

   If the --two-level-tree-map option is supplied, it's interpreted as the
   filename of a file containing a vector<int32> that represents a map from the
   actual pdf-id in the tree, to a pdf-id in a smaller tree- this is obtained
   from two-level tree building.  In this case we interpret the matrix as two
   separate matrices: one matrix with num-cols equal to the number of leaves in
   the first level tree, where each column is the average over all pdf-ids that
   map to that first-level pdf-id; and a second matrix consisting of offsets
   from the level-1 outputs.  And in the expression above, instead of the
   squared l2-norm of the one matrix, we have instead the sum of the squared
   l2-norm of the two matrices.  This will always be <= the l2-norm of the
   original matrix.  We can compute the sum-of-two-matrices l2 norm by
   subtracting a correction term from the l2 norm of the original matrix.
   If a particular 1st-level tree leaf has n 2nd-level tree leaves associated
   with it, we can show that the term we want for the 1st-level tree gets repeated
   n times, while we want it repeated 1 time, so we have to subtract n-1 copies.
   Now, it's more convenient to deal with sums than averages, in terms of the
   CUDA functions we have available, so there is a term (n-1)/n^2 that
   appears in the equation.
*/
void ComputeL2Penalty(const ChainTrainingInfo &info,
                      BaseFloat supervision_weight,
                      const CuMatrixBase<BaseFloat> &nnet_output,
                      BaseFloat *l2_term,
                      CuMatrixBase<BaseFloat> *nnet_output_deriv);

}  // namespace chain
}  // namespace kaldi

#endif  // KALDI_CHAIN_CHAIN_TRAINING_H_

