// nnet3/nnet-adaptation-component.cc

// Copyright      2019  Johns Hopkins University (author: Daniel Povey)

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

#include <iterator>
#include <sstream>
#include <iomanip>
#include "nnet3/nnet-adaptation-component.h"
#include "nnet3/convolution.h"
#include "nnet3/nnet-parse.h"

namespace kaldi {
namespace nnet3 {

AdaptationComponent::AdaptationComponent(): transform_(NULL) { }


AdaptationComponent::AdaptationComponent(
    const AdaptationComponent &other):
    transform_(other.transform_ ? other.transform_->Copy() : NULL),
    accumulate_mode_(other.accumulate_mode_),
    test_mode_(other.test_mode_) { }

std::string AdaptationComponent::Info() const {
  std::ostringstream os;
  // Warning: once models are stored (after Estimate()), this will cause the
  // model params to be printed out, which is not what we want in an info
  // string; we need to find a way to work around this.
  os << Type() << ", dim=" << transform_->Dim()
     << ", accumulate-mode=" << accumulate_mode_
     << ", test-mode=" << test_mode_;
  bool binary = false;
  std::ostringstream transform_os;
  transform_->Write(transform_os, binary);
  if (transform_os.str().size() < 10000) {
    os << ", transform is: " << transform_os.str();
  } else {
    // In future we should find a way of printing transforms without the
    // parameters- maybe a boolean arg to Write().
    os << ", transform is too long to print.";
  }
  return os.str();
}

void AdaptationComponent::InitFromConfig(ConfigLine *cfl) {
  std::string config_file;
  int32 num_classes = -1;
  if (!cfl->GetValue("config-file", &config_file) ||
      !cfl->GetValue("num-classes", &num_classes) ||
      cfl->HasUnusedValues() || num_classes <= 0) {
    KALDI_ERR << "Bad config line: " << cfl->WholeLine();
  }
  delete transform_;  // in case non-NULL.
  bool binary;  // should end up being false.
  Input ki(config_file, &binary);
  transform_ = differentiable_transform::DifferentiableTransform::ReadFromConfig(
      ki.Stream(), num_classes);

  accumulate_mode_ = false;
  test_mode_ = false;
}

void AdaptationComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<AdaptationComponent>");
  WriteToken(os, binary, "<Transform>");
  transform_->Write(os, binary);
  WriteToken(os, binary, "<AccumulateMode>");
  WriteBasicType(os, binary, accumulate_mode_);
  WriteToken(os, binary, "<TestMode>");
  WriteBasicType(os, binary, test_mode_);
  WriteToken(os, binary, "</AdaptationComponent>");
}

void AdaptationComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<AdaptationComponent>", "<Transform>");
  delete transform_;
  transform_ = differentiable_transform::DifferentiableTransform::ReadNew(is, binary);
  ExpectToken(is, binary, "<AccumulateMode>");
  ReadBasicType(is, binary, &accumulate_mode_);
  ExpectToken(is, binary, "<TestMode>");
  ReadBasicType(is, binary, &test_mode_);
  ExpectToken(is, binary, "</AdaptationComponent>");
}


void* AdaptationComponent::Propagate(
    const ClassLabels &class_labels,
    const ComponentPrecomputedIndexes *indexes_in,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  const PrecomputedIndexes *indexes =
      dynamic_cast<const PrecomputedIndexes*>(indexes_in);
  KALDI_ASSERT(class_labels.num_classes == transform_->NumClasses() &&
               class_labels.num_chunks == indexes->num_chunks &&
               in.NumRows() == indexes->num_chunks * indexes->num_t_values);

  if (class_labels.first_t != indexes->first_t ||
      class_labels.t_stride != indexes->t_stride ||
      class_labels.post.size() != size_t(in.NumRows())) {
    ClassLabels modified_labels;
    modified_labels.num_classes = class_labels.num_classes;
    modified_labels.first_t = indexes->first_t;
    modified_labels.t_stride = indexes->t_stride;
    modified_labels.num_chunks = indexes->num_chunks;
    modified_labels.num_spk = class_labels.num_spk;
    modified_labels.post.resize(indexes->num_t_values *
                                indexes->num_chunks);
    ResampleClassLabels(class_labels,
                        &modified_labels);
    return Propagate(modified_labels, indexes_in, in, out);
  }

  if (!test_mode_) {
    Memo *ans = new Memo();
    ans->class_labels = &class_labels;
    ans->minibatch_info = transform_->TrainingForward(
        in, class_labels.num_chunks, class_labels.num_spk,
        class_labels.post, out);
    return ans;
  } else {
    differentiable_transform::SpeakerStatsItf *speaker_stats =
        transform_->GetEmptySpeakerStats();
#if HAVE_CUDA == 1
    if (CuDevice::Instantiate().Enabled()) {
      Matrix<BaseFloat> in_cpu(in), out_cpu(out->NumRows(), out->NumCols());
      transform_->TestingAccumulate(in_cpu, class_labels.post, speaker_stats);
      transform_->TestingForward(in_cpu, *speaker_stats, &out_cpu);
      out->CopyFromMat(out_cpu);
    } else
#endif
    {
      transform_->TestingAccumulate(in.Mat(), class_labels.post, speaker_stats);
      transform_->TestingForward(in.Mat(), *speaker_stats, &(out->Mat()));
    }
    delete speaker_stats;
    return NULL;
  }
}

void AdaptationComponent::StoreStats(
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_value,
    void *memo_in) {
  KALDI_ASSERT(accumulate_mode_ && !test_mode_);
  Memo *memo = static_cast<Memo*>(memo_in);

  int32 final_iter = 0,
      num_chunks = memo->class_labels->num_chunks,
      num_spk = memo->class_labels->num_spk;
  transform_->Accumulate(final_iter, in_value, num_chunks, num_spk,
                         memo->class_labels->post);
}

void AdaptationComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_value,
    const CuMatrixBase<BaseFloat> &out_deriv,
    void *memo_in,
    Component *, // to_update,
    CuMatrixBase<BaseFloat> *in_deriv) const {
  KALDI_ASSERT(memo_in && "Backprop is not supported in test mode.");
  Memo *memo = static_cast<Memo*>(memo_in);

  transform_->TrainingBackward(in_value, out_deriv,
                               memo->class_labels->num_chunks,
                               memo->class_labels->num_spk,
                               memo->class_labels->post,
                               memo->minibatch_info,
                               in_deriv);
}

void AdaptationComponent::SetTestMode(bool test_mode) {
  test_mode_ = test_mode;
  if (test_mode) {
    KALDI_ASSERT(!accumulate_mode_);
  }
}

void AdaptationComponent::Estimate() {
  KALDI_ASSERT(accumulate_mode_);
  int32 final_iter = 0;
  transform_->Estimate(final_iter);
  accumulate_mode_ = false;
}


static void ModifyComputationIo(
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
  KALDI_ASSERT(io->t_step_out == io->t_step_in);
  KALDI_ASSERT(io->reorder_t_in == 1);
}


ComponentPrecomputedIndexes* AdaptationComponent::PrecomputeIndexes(
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
                 modified_output_indexes == output_indexes &&
                 input_indexes == output_indexes);
  }

  PrecomputedIndexes *ans = new PrecomputedIndexes();
  ans->num_chunks = io.num_images;
  ans->first_t = io.start_t_in;
  ans->t_stride = io.t_step_in;
  ans->num_t_values = io.num_t_in;
  KALDI_ASSERT(io.reorder_t_in == 1);
  return ans;
}

AdaptationComponent::PrecomputedIndexes*
AdaptationComponent::PrecomputedIndexes::Copy() const {
  return new PrecomputedIndexes(*this);
}

void AdaptationComponent::PrecomputedIndexes::Write(
    std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<AdaptationComponentPrecomputedIndexes>");
  WriteToken(os, binary, "<Dims>");
  WriteBasicType(os, binary, num_chunks);
  WriteBasicType(os, binary, first_t);
  WriteBasicType(os, binary, t_stride);
  WriteBasicType(os, binary, num_t_values);
  WriteToken(os, binary, "</AdaptationComponentPrecomputedIndexes>");
}

void AdaptationComponent::PrecomputedIndexes::Read(
    std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary,
                       "<AdaptationComponentPrecomputedIndexes>",
                       "<Dims>");
  ReadBasicType(is, binary, &num_chunks);
  ReadBasicType(is, binary, &first_t);
  ReadBasicType(is, binary, &t_stride);
  ReadBasicType(is, binary, &num_t_values);
  ExpectToken(is, binary, "</AdaptationComponentPrecomputedIndexes>");
}




} // namespace nnet3
} // namespace kaldi
