// nnet3/nnet-tdnn-component.h

// Copyright      2017  Johns Hopkins University (author: Daniel Povey)

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

// Note: the code defined here was declared in nnet-convolutional-component.h.

#include <iterator>
#include <sstream>
#include <iomanip>
#include "nnet3/nnet-convolutional-component.h"
#include "nnet3/nnet-computation-graph.h"
#include "nnet3/nnet-parse.h"

namespace kaldi {
namespace nnet3 {


TdnnComponent::TdnnComponent():
    orthonormal_constraint_(0.0),
    joint_l2_regularize_proportion_(0.0),
    use_natural_gradient_(true) { }



TdnnComponent::TdnnComponent(
    const TdnnComponent &other):
    UpdatableComponent(other),  // initialize base-class
    time_offsets_(other.time_offsets_),
    linear_params_(other.linear_params_),
    bias_params_(other.bias_params_),
    orthonormal_constraint_(other.orthonormal_constraint_),
    joint_l2_regularize_proportion_(other.joint_l2_regularize_proportion_),
    use_natural_gradient_(other.use_natural_gradient_),
    preconditioner_in_(other.preconditioner_in_),
    preconditioner_out_(other.preconditioner_out_) {
  Check();
}


void TdnnComponent::Check() const {
  KALDI_ASSERT(linear_params_.NumRows() > 0 &&
               !time_offsets_.empty() &&
               std::set<int32>(time_offsets_.begin(),
                               time_offsets_.end()).size() ==
               time_offsets_.size() &&
               linear_params_.NumCols() % time_offsets_.size() == 0 &&
               (bias_params_.Dim() == 0 ||
                bias_params_.Dim() == linear_params_.NumRows()));
}

std::string TdnnComponent::Info() const {
  std::ostringstream stream;
  stream << UpdatableComponent::Info();
  if (orthonormal_constraint_ != 0.0)
    stream << ", orthonormal-constraint=" << orthonormal_constraint_;
  if (joint_l2_regularize_proportion_ != 0.0)
    stream << ", joint-l2-regularize-proportion="
           << joint_l2_regularize_proportion_;
  stream << ", time-offsets=";
  for (size_t i = 0; i < time_offsets_.size(); i++) {
    if (i != 0) stream << ',';
    stream << time_offsets_[i];
  }
  PrintParameterStats(stream, "linear-params", linear_params_,
                      false, // include_mean
                      true, // include_row_norms
                      true, // include_column_norms
                      GetVerboseLevel() >= 2); // include_singular_values
  if (bias_params_.Dim() == 0) {
    stream << ", has-bias=false";
  } else {
    PrintParameterStats(stream, "bias", bias_params_, true);
  }
  if (!use_natural_gradient_) {
    stream << ", use-natural-gradient=false";
  } else {
    stream << ", rank-in=" << preconditioner_in_.GetRank()
           << ", rank-out=" << preconditioner_out_.GetRank()
           << ", num-samples-history=" << preconditioner_in_.GetNumSamplesHistory()
           << ", update-period=" << preconditioner_in_.GetUpdatePeriod()
           << ", alpha=" << preconditioner_in_.GetAlpha();
  }
  return stream.str();
}


void TdnnComponent::InitFromConfig(ConfigLine *cfl) {
  // 1. Config values inherited from UpdatableComponent.
  InitLearningRatesFromConfig(cfl);

  // 2. Structural config values
  std::string time_offsets;

  int32 input_dim = -1, output_dim = -1;

  bool ok = cfl->GetValue("time-offsets", &time_offsets) &&
      cfl->GetValue("input-dim", &input_dim) &&
      cfl->GetValue("output-dim", &output_dim);
  if (!ok || input_dim <= 0 || output_dim <= 0 ||
      !SplitStringToIntegers(time_offsets, ",", false, &time_offsets_) ||
      time_offsets_.empty()) {
    KALDI_ERR << "Bad initializer: there is a problem with "
        "time-offsets, input-dim or output-dim (not defined?): "
        << cfl->WholeLine();
  }

  if (std::set<int32>(time_offsets_.begin(),
                      time_offsets_.end()).size() != time_offsets_.size()) {
    KALDI_ERR << "Bad initializer: repeated time-offsets: "
              << cfl->WholeLine();
  }

  // 3. Parameter-initialization configs, "has-bias", and
  // "orthonormal-constraint" and "joint-l2-regularize-proportion".
  orthonormal_constraint_ = 0.0;
  joint_l2_regularize_proportion_ = 0.0;
  BaseFloat param_stddev = -1, bias_mean = 0.0, bias_stddev = 1.0;
  bool use_bias = true;
  cfl->GetValue("param-stddev", &param_stddev);
  cfl->GetValue("bias-stddev", &bias_stddev);
  cfl->GetValue("bias-mean", &bias_mean);
  cfl->GetValue("use-bias", &use_bias);
  cfl->GetValue("orthonormal-constraint", &orthonormal_constraint_);
  cfl->GetValue("joint-l2-regularize-proportion",
                &joint_l2_regularize_proportion_);
  if (param_stddev < 0.0) {
    param_stddev = 1.0 / sqrt(input_dim * time_offsets_.size());
  }
  if (joint_l2_regularize_proportion_ < 0.0 ||
      joint_l2_regularize_proportion_ > 1.0) {
    KALDI_ERR << "Invalid value joint-l2-regularize-proportion="
              << joint_l2_regularize_proportion_;
  }

  // initialize the parameters.
  linear_params_.Resize(output_dim,
                        input_dim * time_offsets_.size());
  linear_params_.SetRandn();
  linear_params_.Scale(param_stddev);

  if (use_bias) {
    bias_params_.Resize(output_dim);
    bias_params_.SetRandn();
    bias_params_.Scale(bias_stddev);
    bias_params_.Add(bias_mean);
  } else {
    bias_params_.Resize(0);
  }

  // 4. Natural-gradient related configs.
  use_natural_gradient_ = true;
  int32 rank_out = -1, rank_in = -1;
  BaseFloat alpha_out = 4.0, alpha_in = 4.0,
      num_samples_history = 2000.0;
  cfl->GetValue("use-natural-gradient", &use_natural_gradient_);
  cfl->GetValue("rank-in", &rank_in);
  cfl->GetValue("rank-out", &rank_out);
  cfl->GetValue("alpha-in", &alpha_in);
  cfl->GetValue("alpha-out", &alpha_out);
  cfl->GetValue("num-samples-history", &num_samples_history);

  int32 spliced_input_dim =
      input_dim * static_cast<int32>(time_offsets_.size());
  if (rank_in < 0)
    rank_in = std::min<int32>(20, (spliced_input_dim + 1) / 2);
  preconditioner_in_.SetRank(rank_in);
  if (rank_out < 0)
    rank_out = std::min<int32>(80, (output_dim + 1) / 2);
  preconditioner_out_.SetRank(rank_out);
  preconditioner_in_.SetNumSamplesHistory(num_samples_history);
  preconditioner_out_.SetNumSamplesHistory(num_samples_history);

  preconditioner_in_.SetAlpha(alpha_in);
  preconditioner_out_.SetAlpha(alpha_out);

  preconditioner_in_.SetUpdatePeriod(4);
  preconditioner_out_.SetUpdatePeriod(4);
}

void* TdnnComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes_in,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  const PrecomputedIndexes *indexes =
      dynamic_cast<const PrecomputedIndexes*>(indexes_in);
  KALDI_ASSERT(indexes != NULL);

  if (bias_params_.Dim() != 0)
    out->CopyRowsFromVec(bias_params_);
  // if bias_params_.Dim() == 0 we don't need to zero 'out' at
  // this point because in that case we set the flag kPropagateAdds,
  // so the calling code knows that the Propagate function *adds to*
  // the 'out' matrix, so it should (typicaly) be zeroed before calling
  // Propagate().

  KALDI_ASSERT(indexes->row_offsets.size() == time_offsets_.size());

  int32 num_offsets = time_offsets_.size(),
      input_dim = InputDim();
  for (int32 i = 0; i < num_offsets; i++) {
    CuSubMatrix<BaseFloat> in_part = GetInputPart(in, out->NumRows(),
                                                  indexes->row_stride,
                                                  indexes->row_offsets[i]);
    CuSubMatrix<BaseFloat> linear_params_part(linear_params_,
                                              0, linear_params_.NumRows(),
                                              i * input_dim, input_dim);
    out->AddMatMat(1.0, in_part, kNoTrans, linear_params_part, kTrans, 1.0);
  }
  return NULL;
}

void TdnnComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *indexes_in,
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &, // out_value
    const CuMatrixBase<BaseFloat> &out_deriv,
    void*, // memo
    Component *to_update_in,
    CuMatrixBase<BaseFloat> *in_deriv) const {
  const PrecomputedIndexes *indexes =
      dynamic_cast<const PrecomputedIndexes*>(indexes_in);
  KALDI_ASSERT(indexes != NULL &&
               indexes->row_offsets.size() == time_offsets_.size());
  int32 num_offsets = time_offsets_.size(),
      input_dim = InputDim();

  if (in_deriv != NULL) {
    // Propagate the derivatives back to the input data.
    for (int32 i = 0; i < num_offsets; i++) {
      CuSubMatrix<BaseFloat> in_deriv_part =
          GetInputPart(*in_deriv, out_deriv.NumRows(),
                       indexes->row_stride, indexes->row_offsets[i]);
      CuSubMatrix<BaseFloat> linear_params_part(linear_params_,
                                                0, linear_params_.NumRows(),
                                                i * input_dim, input_dim);
      // note: this component has the property kBackpropAdds, which is why the
      // final 1.0 is there in the following call (otherwise we'd have to zero
      // *in_deriv first).
      in_deriv_part.AddMatMat(1.0, out_deriv, kNoTrans,
                              linear_params_part, kNoTrans, 1.0);
    }
  }

  if (to_update_in != NULL) {
    TdnnComponent *to_update =
        dynamic_cast<TdnnComponent*>(to_update_in);
    KALDI_ASSERT(to_update != NULL);

    if (to_update->learning_rate_ == 0.0)
      return;

    if (to_update->is_gradient_ || !to_update->use_natural_gradient_)
      to_update->UpdateSimple(*indexes, in_value, out_deriv);
    else
      to_update->UpdateNaturalGradient(*indexes, in_value, out_deriv);
  }
}

void TdnnComponent::UpdateSimple(
    const PrecomputedIndexes &indexes,
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_deriv) {

  if (bias_params_.Dim() != 0)
    bias_params_.AddRowSumMat(learning_rate_, out_deriv);

  int32 input_dim = in_value.NumCols(),
      num_offsets = time_offsets_.size();
  for (int32 i = 0; i < num_offsets; i++) {
    CuSubMatrix<BaseFloat> in_value_part =
        GetInputPart(in_value, out_deriv.NumRows(),
                     indexes.row_stride,
                     indexes.row_offsets[i]);
    CuSubMatrix<BaseFloat> linear_params_part(linear_params_,
                                              0, linear_params_.NumRows(),
                                              i * input_dim, input_dim);
    linear_params_part.AddMatMat(learning_rate_, out_deriv, kTrans,
                                 in_value_part, kNoTrans, 1.0);
  }
}

void TdnnComponent::UpdateNaturalGradient(
    const PrecomputedIndexes &indexes,
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_deriv) {

  int32 num_offsets = time_offsets_.size(),
      num_rows = out_deriv.NumRows(),
      input_dim = in_value.NumCols(),
      spliced_input_dim = num_offsets * input_dim,
      augmented_input_dim =
        spliced_input_dim + (bias_params_.Dim() != 0 ? 1 : 0);

  // in_value_temp is the fully spliced input with a column of ones appended to
  // it.
  CuMatrix<BaseFloat> in_value_temp(num_rows,
                                    augmented_input_dim);
  if (bias_params_.Dim() != 0) {
    // set the last column of in_value_temp to 1.0
    in_value_temp.Range(0, num_rows, spliced_input_dim, 1).Set(1.0);
  }

  for (int32 i = 0; i < num_offsets; i++) {
    CuSubMatrix<BaseFloat> in_value_temp_part(in_value_temp,
                                              0, num_rows,
                                              i * input_dim, input_dim),
        in_value_part = GetInputPart(in_value,
                                     num_rows,
                                     indexes.row_stride,
                                     indexes.row_offsets[i]);
    in_value_temp_part.CopyFromMat(in_value_part);
  }

  CuMatrix<BaseFloat> out_deriv_temp(out_deriv);

  // These "scale" values get will get multiplied into the learning rate (faster
  // than having the matrices scaled inside the preconditioning code).
  BaseFloat in_scale, out_scale;

  preconditioner_in_.PreconditionDirections(&in_value_temp, &in_scale);
  preconditioner_out_.PreconditionDirections(&out_deriv_temp, &out_scale);

  // "scale" is a scaling factor coming from the PreconditionDirections calls
  // (it's faster to have them output a scaling factor than to have them scale
  // their outputs).
  BaseFloat scale = in_scale * out_scale,
      local_lrate = scale * learning_rate_;

  if (bias_params_.Dim() != 0) {
    // this "precon_ones" is what happens to the vector of 1's representing
    // offsets, after multiplication by the preconditioner.
    CuVector<BaseFloat> precon_ones(num_rows);
    precon_ones.CopyColFromMat(in_value_temp, spliced_input_dim);
    bias_params_.AddMatVec(local_lrate, out_deriv_temp, kTrans,
                           precon_ones, 1.0);
  }

  CuSubMatrix<BaseFloat> in_value_precon_part(in_value_temp,
                                              0, num_rows,
                                              0, spliced_input_dim);

  linear_params_.AddMatMat(local_lrate, out_deriv_temp, kTrans,
                           in_value_precon_part, kNoTrans, 1.0);
}

void TdnnComponent::ReorderIndexes(
    std::vector<Index> *input_indexes,
    std::vector<Index> *output_indexes) const {
  using namespace time_height_convolution;

  // The following figures out a regular structure for the input and
  // output indexes, in case there were gaps (which is unlikely in typical
  // situations).
  ConvolutionComputationIo io;
  GetComputationIo(*input_indexes, *output_indexes, &io);
  ModifyComputationIo(&io);

  std::vector<Index> modified_input_indexes,
      modified_output_indexes;
  // The following call ensures that 'modified_input_indexes' and
  // 'modified_output_indexes' have the required ordering (where t has the
  // largest stride and each (n,x) pair is repeated for each 't' value), as well
  // as doing padding (setting t values to kNoTime where it had to insert
  // elements to ensure regular structure).
  GetIndexesForComputation(io, *input_indexes, *output_indexes,
                           &modified_input_indexes,
                           &modified_output_indexes);

  // It will be quite rare that this function actually changes
  // 'input_indexes' or 'output_indexes', because in most cases,
  // the indexes will already have the required structure and
  // ordering.
  input_indexes->swap(modified_input_indexes);
  output_indexes->swap(modified_output_indexes);
}

void TdnnComponent::Write(std::ostream &os, bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write opening tag and learning rate.
  WriteToken(os, binary, "<TimeOffsets>");
  WriteIntegerVector(os, binary, time_offsets_);
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "<OrthonormalConstraint>");
  WriteBasicType(os, binary, orthonormal_constraint_);
  if (joint_l2_regularize_proportion_ != 0.0) {
    WriteToken(os, binary, "<JointL2RegularizeProportion>");
    WriteBasicType(os, binary, joint_l2_regularize_proportion_);
  }
  WriteToken(os, binary, "<UseNaturalGradient>");
  WriteBasicType(os, binary, use_natural_gradient_);
  int32 rank_in = preconditioner_in_.GetRank(),
      rank_out = preconditioner_out_.GetRank();
  BaseFloat alpha_in = preconditioner_in_.GetAlpha(),
      alpha_out = preconditioner_out_.GetAlpha(),
      num_samples_history = preconditioner_in_.GetNumSamplesHistory();
  WriteToken(os, binary, "<NumSamplesHistory>");
  WriteBasicType(os, binary, num_samples_history);
  WriteToken(os, binary, "<AlphaInOut>");
  WriteBasicType(os, binary, alpha_in);
  WriteBasicType(os, binary, alpha_out);
  WriteToken(os, binary, "<RankInOut>");
  WriteBasicType(os, binary, rank_in);
  WriteBasicType(os, binary, rank_out);
  WriteToken(os, binary, "</TdnnComponent>");
}

void TdnnComponent::Read(std::istream &is, bool binary) {
  std::string token = ReadUpdatableCommon(is, binary);
  ExpectToken(is, binary, "<TimeOffsets>");
  ReadIntegerVector(is, binary, &time_offsets_);
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  ExpectToken(is, binary, "<OrthonormalConstraint>");
  ReadBasicType(is, binary, &orthonormal_constraint_);
  if (PeekToken(is, binary) == 'J') {
    ExpectToken(is, binary, "<JointL2RegularizeProportion>");
    ReadBasicType(is, binary, &joint_l2_regularize_proportion_);
  } else {
    joint_l2_regularize_proportion_ = 0.0;
  }
  ExpectToken(is, binary, "<UseNaturalGradient>");
  ReadBasicType(is, binary, &use_natural_gradient_);
  int32 rank_in,  rank_out;
  BaseFloat alpha_in, alpha_out,
      num_samples_history;
  ExpectToken(is, binary, "<NumSamplesHistory>");
  ReadBasicType(is, binary, &num_samples_history);
  ExpectToken(is, binary, "<AlphaInOut>");
  ReadBasicType(is, binary, &alpha_in);
  ReadBasicType(is, binary, &alpha_out);
  preconditioner_in_.SetAlpha(alpha_in);
  preconditioner_out_.SetAlpha(alpha_out);
  ExpectToken(is, binary, "<RankInOut>");
  ReadBasicType(is, binary, &rank_in);
  ReadBasicType(is, binary, &rank_out);
  preconditioner_in_.SetRank(rank_in);
  preconditioner_out_.SetRank(rank_out);
  preconditioner_in_.SetNumSamplesHistory(num_samples_history);
  preconditioner_out_.SetNumSamplesHistory(num_samples_history);
  // the update periods are not configurable.
  preconditioner_in_.SetUpdatePeriod(4);
  preconditioner_out_.SetUpdatePeriod(4);
  ExpectToken(is, binary, "</TdnnComponent>");
  Check();
}

void TdnnComponent::GetInputIndexes(
    const MiscComputationInfo &misc_info,
    const Index &output_index,
    std::vector<Index> *desired_indexes) const {
  KALDI_ASSERT(output_index.t != kNoTime);
  size_t size = time_offsets_.size();
  desired_indexes->resize(size);
  for (size_t i = 0; i < size; i++) {
    (*desired_indexes)[i].n = output_index.n;
    (*desired_indexes)[i].t = output_index.t + time_offsets_[i];
    (*desired_indexes)[i].x = output_index.x;
  }
}


bool TdnnComponent::IsComputable(
    const MiscComputationInfo &misc_info,
    const Index &output_index,
    const IndexSet &input_index_set,
    std::vector<Index> *used_inputs) const {
  KALDI_ASSERT(output_index.t != kNoTime);
  size_t size = time_offsets_.size();
  Index index(output_index);

  if (used_inputs != NULL) {
    used_inputs->clear();
    used_inputs->reserve(size);
  }
  for (size_t i = 0; i < size; i++) {
    index.t = output_index.t + time_offsets_[i];
    if (input_index_set(index)) {
      if (used_inputs != NULL) {
        // This input index is available.
        used_inputs->push_back(index);
      }
    } else {
      return false;
    }
  }
  return true;
}

// static
CuSubMatrix<BaseFloat> TdnnComponent::GetInputPart(
      const CuMatrixBase<BaseFloat> &input_matrix,
      int32 num_output_rows,
      int32 row_stride,
      int32 row_offset) {
  KALDI_ASSERT(row_offset >= 0 && row_stride >= 1 &&
               input_matrix.NumRows() >=
               row_offset + (row_stride * num_output_rows) - (row_stride - 1));
  // constructor takes args: (data, num_rows, num_cols, stride).
  return CuSubMatrix<BaseFloat>(
      input_matrix.Data() + input_matrix.Stride() * row_offset,
      num_output_rows,
      input_matrix.NumCols(),
      input_matrix.Stride() * row_stride);
}

void TdnnComponent::ModifyComputationIo(
    time_height_convolution::ConvolutionComputationIo *io) {
  if (io->t_step_out == 0) {
    // the 't_step' values may be zero if there was only one (input or output)
    // index so the time-stride could not be determined.  This code fixes them
    // up in that case.  (If there was only one value, the stride is a
    // don't-care actually).
    if (io->t_step_in == 0)
      io->t_step_in = 1;
    io->t_step_out = io->t_step_in;
  }
  // At this point the t_step_{in,out} values will be nonzero.
  KALDI_ASSERT(io->t_step_out % io->t_step_in == 0);
  // The following affects the ordering of the input indexes; it allows us to
  // reshape the input matrix in the way that we need to, in cases where there
  // is subsampling.  See the explanation where the variable was declared in
  // class ConvolutionComputationIo.
  io->reorder_t_in = io->t_step_out / io->t_step_in;

  // make sure that num_t_in is a multiple of io->reorder_t_in by rounding up.
  int32 n = io->reorder_t_in;
  io->num_t_in = n * ((io->num_t_in + n - 1) / n);
}

ComponentPrecomputedIndexes* TdnnComponent::PrecomputeIndexes(
      const MiscComputationInfo &misc_info,
      const std::vector<Index> &input_indexes,
      const std::vector<Index> &output_indexes,
      bool need_backprop) const {
  using namespace time_height_convolution;
  // The following figures out a regular structure for the input and
  // output indexes, in case there were gaps (which is unlikely in typical
  // situations).
  ConvolutionComputationIo io;
  GetComputationIo(input_indexes, output_indexes, &io);
  ModifyComputationIo(&io);

  if (RandInt(0, 10) == 0) {
    // Spot check that the provided indexes have the required properties;
    // this is like calling this->ReorderIndexes() and checking that it
    // doesn't change anything.
    std::vector<Index> modified_input_indexes,
        modified_output_indexes;
    GetIndexesForComputation(io, input_indexes, output_indexes,
                             &modified_input_indexes,
                             &modified_output_indexes);
    KALDI_ASSERT(modified_input_indexes == input_indexes &&
                 modified_output_indexes == output_indexes);
  }


  PrecomputedIndexes *ans = new PrecomputedIndexes();
  ans->row_stride = io.reorder_t_in;
  int32 num_offsets = time_offsets_.size();
  ans->row_offsets.resize(num_offsets);
  for (int32 i = 0; i < num_offsets; i++) {
    // For each offset, work out which row of the input has the same t value as
    // the first t value in the output plus that offset.  That becomes the start
    // row of the corresponding sub-part of the input.
    int32 time_offset = time_offsets_[i],
        required_input_t = io.start_t_out + time_offset,
        input_t = (required_input_t - io.start_t_in) / io.t_step_in;

    KALDI_ASSERT(required_input_t == io.start_t_in + io.t_step_in * input_t);
    // input_t is a kind of normalized time offset in the input, relative to the
    // first 't' value in the input and divided by the t-step in the input, so
    // it's the numbering "as if" the input 't' values were numbered from 0,1,2.
    // To turn input_t into an input row we need to take account of 'reorder_t_in'.
    // If this is 1 then the input row is input_t times io.num_images.
    // Otherwise it's a little more complicated and to understand it you should
    // read the comment where 'reorder_t_in' is declared in convolution.h.
    // Briefly: the part that is an integer multiple of 'reorder_t_in' gets
    // multiplied by io.num_images; the remainder does not.

    int32 n = io.reorder_t_in,
        input_t_multiple = n * (input_t / n), input_t_remainder = input_t % n;
    // note: input_t == input_t_multiple + input_t_remainder .
    int32 input_row_offset = input_t_multiple * io.num_images +
        input_t_remainder;
    ans->row_offsets[i] = input_row_offset;
  }
  return ans;
}

void TdnnComponent::Scale(BaseFloat scale) {
  if (scale == 0.0) {
    linear_params_.SetZero();
    bias_params_.SetZero();
  } else {
    linear_params_.Scale(scale);
    bias_params_.Scale(scale);
  }
}

void TdnnComponent::Add(BaseFloat alpha,
                        const Component &other_in) {
  const TdnnComponent *other =
      dynamic_cast<const TdnnComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  linear_params_.AddMat(alpha, other->linear_params_);
  if (bias_params_.Dim() != 0)
    bias_params_.AddVec(alpha, other->bias_params_);
}

void TdnnComponent::PerturbParams(BaseFloat stddev) {
  CuMatrix<BaseFloat> temp_mat(linear_params_.NumRows(),
                               linear_params_.NumCols(), kUndefined);
  temp_mat.SetRandn();
  linear_params_.AddMat(stddev, temp_mat);
  if (bias_params_.Dim() != 0) {
    CuVector<BaseFloat> temp_vec(bias_params_.Dim(), kUndefined);
    temp_vec.SetRandn();
    bias_params_.AddVec(stddev, temp_vec);
  }
}

BaseFloat TdnnComponent::DotProduct(
    const UpdatableComponent &other_in) const {
  const TdnnComponent *other =
      dynamic_cast<const TdnnComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  BaseFloat ans = TraceMatMat(linear_params_, other->linear_params_, kTrans);
  if (bias_params_.Dim() != 0)
    ans += VecVec(bias_params_, other->bias_params_);
  return ans;
}

int32 TdnnComponent::NumParameters() const {
  // note: bias_param_.Dim() may actually be zero.
  return linear_params_.NumRows() * linear_params_.NumCols() +
      bias_params_.Dim();
}

void TdnnComponent::Vectorize(
    VectorBase<BaseFloat> *params) const {
  KALDI_ASSERT(params->Dim() == NumParameters());
  int32 linear_size = linear_params_.NumRows() * linear_params_.NumCols(),
      bias_size = bias_params_.Dim();
  params->Range(0, linear_size).CopyRowsFromMat(linear_params_);
  if (bias_size != 0)
    params->Range(linear_size, bias_size).CopyFromVec(bias_params_);
}

void TdnnComponent::UnVectorize(
    const VectorBase<BaseFloat> &params) {
  KALDI_ASSERT(params.Dim() == NumParameters());
  int32 linear_size = linear_params_.NumRows() * linear_params_.NumCols(),
      bias_size = bias_params_.Dim();
  linear_params_.CopyRowsFromVec(params.Range(0, linear_size));
  if (bias_size != 0)
    bias_params_.CopyFromVec(params.Range(linear_size, bias_size));
}

void TdnnComponent::FreezeNaturalGradient(bool freeze) {
  preconditioner_in_.Freeze(freeze);
  preconditioner_out_.Freeze(freeze);
}

TdnnComponent::PrecomputedIndexes*
TdnnComponent::PrecomputedIndexes::Copy() const {
  return new PrecomputedIndexes(*this);
}

void TdnnComponent::PrecomputedIndexes::Write(
    std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<TdnnComponentPrecomputedIndexes>");
  WriteToken(os, binary, "<RowStride>");
  WriteBasicType(os, binary, row_stride);
  WriteToken(os, binary, "<RowOffsets>");
  WriteIntegerVector(os, binary, row_offsets);
  WriteToken(os, binary, "</TdnnComponentPrecomputedIndexes>");
}

void TdnnComponent::PrecomputedIndexes::Read(
    std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary,
                       "<TdnnComponentPrecomputedIndexes>",
                       "<RowStride>");
  ReadBasicType(is, binary, &row_stride);
  ExpectToken(is, binary, "<RowOffsets>");
  ReadIntegerVector(is, binary, &row_offsets);
  ExpectToken(is, binary, "</TdnnComponentPrecomputedIndexes>");
}


// This utility function is used to propagate the derivatives of the squared l2
// norm of the matrix product:
//     -l2_regularize_factor * ||A B||_2^2
// back to A and B themselves.  It adds the derivative of this quantitity
// w.r.t. A and B respectively, to "A_deriv" and "B_deriv".
//
// The thing we're taking the derivative of can be written as:
//   -l2_regularize_factor * tr( A B (A B)^T )
// = -l2_regularize_factor * tr( A B B^T A^T )
// = -l2_regularize_factor * tr( A^T A B B^T )
// If we're using a convention where the derivative of a matrix
// w.r.t. A is always of the same dimension as A (i.e. not its transpose),
// then the derivative of this w.r.t. A is:
//  df/dA =  -2.0 * l2_regularize_factor *  A B B^T
// and the derivative w.r.t. B is:
//  df/dB =  -2.0 * l2_regularize_factor *  A^T A B
// This function adds those two derivatives to 'A_deriv' and 'B_deriv'
// respectively.
// Just for background: if we were doing regular L2 regularization on A and B
// (i.e. not on their product), the expressions for the derivatives above would
// be the same except with the symmetric factors (A^T A) and (B B^T) taken out.
static void ApplyProductL2Regularization(
    BaseFloat l2_regularize_factor,
    const CuMatrixBase<BaseFloat> &A,
    const CuMatrixBase<BaseFloat> &B,
    CuMatrixBase<BaseFloat> *A_deriv,
    CuMatrixBase<BaseFloat> *B_deriv) {


  int32 bottleneck_dim = B.NumRows();  // and A.NumCols().
  // BBT is B B^T.
  CuMatrix<BaseFloat> BBT(bottleneck_dim, bottleneck_dim);


  { // The following block adds the derivative w.r.t. A to A_deriv.
    // the following two commands should be equivalent to (but hopefully
    // slightly faster than):
    //  BBT.SymAddMat2(1.0, B, kNoTrans, B, kTrans, 0.0);
    BBT.SymAddMat2(1.0, B, kNoTrans, 0.0);
    BBT.CopyLowerToUpper();
    // note: in the next statement it doesn't matter whether we give BBT
    // transposed or not: in principle we could tune that for speed.
    A_deriv->AddMatMat(-2.0 * l2_regularize_factor, A, kNoTrans, BBT, kNoTrans, 1.0);
  }
  { // The following block adds the derivative w.r.t. B to B_deriv.  ATA is A^T
    // A.  We re-use the no-longer-needed matrix BBT for storage.
    CuMatrix<BaseFloat> &ATA(BBT);
    // the following two commands should be equivalent to (but hopefully
    // slightly faster than):
    //  ATA.SymAddMat2(1.0, A, kTrans, A, kNoTrans, 0.0);
    ATA.SymAddMat2(1.0, A, kTrans, 0.0);
    ATA.CopyLowerToUpper();
    // note: in the next statement it doesn't matter whether we give ATA
    // transposed or not: in principle we could tune that for speed.
    B_deriv->AddMatMat(-2.0 * l2_regularize_factor, ATA, kNoTrans, B, kNoTrans, 1.0);
  }
}

void TdnnComponent::ApplyBottleneckL2Regularization(
    BaseFloat l2_regularize_factor,
    const TdnnComponent &next_component,
    TdnnComponent *this_delta_component,
    TdnnComponent *next_delta_component) const {
  KALDI_ASSERT(bias_params_.Dim() == 0 &&
               SameDim(linear_params_, this_delta_component->linear_params_) &&
               SameDim(next_component.linear_params_,
                       next_delta_component->linear_params_) &&
               orthonormal_constraint_ == 0.0 &&
               next_component.orthonormal_constraint_ == 0.0 &&
               l2_regularize_ != 0.0 && next_component.l2_regularize_ != 0.0 &&
               l2_regularize_ == next_component.l2_regularize_ &&
               OutputDim() == next_component.InputDim());

  // The num-rows of this_params is OutputDim() == next_component.InputDim().
  // This will become B in ApplyProductL2Regularization(); it's on the right
  // because it's applied to the parameters first.
  const CuMatrix<BaseFloat> &this_params(linear_params_);
  CuMatrix<BaseFloat> *this_deriv(&(this_delta_component->linear_params_));

  // we don't need to interrogate this->l2_regularize_ or this->learning_rate_
  // because they have already been made factors of 'l2_regularize_factor'.
  if (next_component.linear_params_.NumCols() == next_component.InputDim()) {
    // If the next component has no splicing over time, then we don't need
    // to reorganize its parameter matrix's blocks.
    const CuMatrix<BaseFloat> &next_params(next_component.linear_params_);
    CuMatrix<BaseFloat> *next_deriv(&(next_delta_component->linear_params_));
    ApplyProductL2Regularization(l2_regularize_factor, next_params, this_params,
                                 next_deriv, this_deriv);
  } else {
    // The next component has splicing over time.  Its linear_params_ matrix
    // would have num-cols equal to the bottleneck-dim times the number of
    // spliced frames it accepts.  In order to view the sequence of two
    // components as a matrix multiplication we have to make that parameter
    // matrix's num-cols equal to the bottleneck-dim, and increase num-rows by a
    // factor equal to the amount of splicing.  The product matrix would now
    // have a possibly strange interpretation, with blocks that could be indexed
    // by the time-offsets on the two components.
    int32 input_dim = next_component.InputDim(),
        output_dim = next_component.OutputDim(),
        num_offsets =  next_component.linear_params_.NumCols() / input_dim;
    // We'll use two blocks of the following for the parameters and (derivative
    // w.r.t. parameters) of the next component.  We could find a way to do it
    // in-place but this is clearer; we'll reorganize it if the speed of this
    // seems to be a problem
    CuMatrix<BaseFloat> storage(2 * num_offsets * output_dim,
                                input_dim, kUndefined);
    CuSubMatrix<BaseFloat> next_component_params =
        storage.RowRange(0, num_offsets * output_dim);
    CuSubMatrix<BaseFloat> next_component_deriv =
        storage.RowRange(num_offsets * output_dim,
                         num_offsets * output_dim);
    // Copy the blocks of the parameters to temporary storage (rearranging
    // blocks).
    for (int32 o = 0; o < num_offsets; o++) {
      CuSubMatrix<BaseFloat> params_block_dest(next_component_params,
                                               o * output_dim, output_dim,
                                               0, input_dim),
          params_block_src(next_component.linear_params_,
                           0, output_dim, o * input_dim, input_dim);
      params_block_dest.CopyFromMat(params_block_src);
    }
    next_component_deriv.SetZero();
    ApplyProductL2Regularization(l2_regularize_factor,
                                 next_component_params, this_params,
                                 &next_component_deriv, this_deriv);
    // Add the blocks of the parameter derivatives to
    // next_delta_component->linear_params_.
    for (int32 o = 0; o < num_offsets; o++) {
      CuSubMatrix<BaseFloat> deriv_block_src(next_component_deriv,
                                             o * output_dim, output_dim,
                                             0, input_dim),
          deriv_block_dest(next_delta_component->linear_params_,
                           0, output_dim, o * input_dim, input_dim);
      deriv_block_dest.AddMat(1.0, deriv_block_src);
    }
  }

  // Appply l2 on the bias parameters.
  if (next_delta_component->bias_params_.Dim() != 0) {
    next_delta_component->bias_params_.AddVec(-2.0 * l2_regularize_factor,
                                              next_component.bias_params_);
  }
}



} // namespace nnet3
} // namespace kaldi
