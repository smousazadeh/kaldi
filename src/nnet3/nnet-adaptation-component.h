// nnet3/nnet-adaptation-component.h

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

#ifndef KALDI_NNET3_NNET_ADAPTATION_COMPONENT_H_
#define KALDI_NNET3_NNET_ADAPTATION_COMPONENT_H_

#include "nnet3/nnet-common.h"
#include "nnet3/nnet-component-itf.h"
#include "adapt/differentiable-transform-itf.h"
#include <iostream>

namespace kaldi {
namespace nnet3 {

/// @file  nnet-adaptation-component.h
///
/// This file contains the declaration of class AdaptationComponent, which is
/// an nnet3 wrapper of the interface declared in ../adapt/differentiable-transform.h.



/**

   The config line accepts lines like:
     num-classes=200
     config-file=foo/bar.txt
   where foo/bar.txt is the config file used to create the object of type
   DifferentiableTransform.
 */
class AdaptationComponent: public Component {
 public:

  AdaptationComponent();

  AdaptationComponent(const AdaptationComponent &other);


  virtual int32 InputDim() const { return transform_->Dim(); }
  virtual int32 OutputDim() const { return transform_->Dim(); }

  virtual std::string Info() const;
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual std::string Type() const { return "AdaptationComponent"; }
  virtual int32 Properties() const {
    return kReordersIndexes|kBackpropAdds|kBackpropNeedsInput|
        (accumulate_mode_ ? kStoresStats : 0)| kUsesMemo;
  }

  virtual void* Propagate(const ClassLabels &class_labels,
                          const ComponentPrecomputedIndexes *indexes,
                          const CuMatrixBase<BaseFloat> &in,
                          CuMatrixBase<BaseFloat> *out) const;

  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                          CuMatrixBase<BaseFloat> *out) const {
    KALDI_ERR << "This nnet requires the ClassLabels object... likely "
        "you missed a command-line option.";  // TODO: make more specific.
  }

  virtual void StoreStats(const CuMatrixBase<BaseFloat> &in_value,
                          const CuMatrixBase<BaseFloat> &out_value,
                          void *memo);

  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *, // to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Read(std::istream &is, bool binary); // This Read function
  // requires that the Component has the correct type.

  /// Write component to stream
  virtual void Write(std::ostream &os, bool binary) const;
  virtual Component* Copy() const { return new AdaptationComponent(*this); }

  virtual ComponentPrecomputedIndexes* PrecomputeIndexes(
      const MiscComputationInfo &misc_info,
      const std::vector<Index> &input_indexes,
      const std::vector<Index> &output_indexes,
      bool need_backprop) const;

  // Some functions that are specific to this class.

  // You call SetAccumulateMode after the neural net is trained, if you want it to
  // accumulate stats for the adaptation model (needed in test mode.)
  void SetAccumulateMode(bool b) { accumulate_mode_ = b; }


  // Set test mode to the provided value.  If true, it will also set
  // accumulate-mode to false.
  void SetTestMode(bool test_mode);

  // You call Estimate() after accumulating the stats by running the model
  // after setting SetAccumulateMode to true.  It will also set
  // accumulate_mode_ to false.
  void Estimate();

  virtual void DeleteMemo(void *memo) const {
    delete static_cast<Memo*>(memo);
  }

  class PrecomputedIndexes: public ComponentPrecomputedIndexes {
   public:
    PrecomputedIndexes() { }
    PrecomputedIndexes(const PrecomputedIndexes &other) = default;

    virtual PrecomputedIndexes *Copy() const;
    virtual void Write(std::ostream &os, bool binary) const;
    virtual void Read(std::istream &os, bool binary);
    virtual std::string Type() const {
      return "AdaptationComponentPrecomputedIndexes";
    }
    virtual ~PrecomputedIndexes() { }


    // The number of chunks (i.e. number of distinct 'n' values) in the input/output
    // indexes (the input and output indexes must be the same).
    int32 num_chunks;
    // The first 't' value in the input indexes (e.g. 0).  The expected ordering
    // is: all 'n' values for the first t; then all 'n' values for the second t;
    // and so on.
    int32 first_t;
    // The stride of the 't' values; will usually be 1 or 3.
    int32 t_stride;
    // The number of distinct 't' values; this times num_chunks is the number of
    // rows in the input and output.
    int32 num_t_values;
  };

 private:

  struct Memo {
    const ClassLabels *class_labels;
    // 'minibatch_info' is what TrainingForward() of class DifferentiableTransform
    // returns; it's needed in the backprop.
    differentiable_transform::MinibatchInfoItf *minibatch_info;
    Memo(): class_labels(NULL), minibatch_info(NULL) {}
    ~Memo() { delete class_labels; delete minibatch_info; }
  };

  differentiable_transform::DifferentiableTransform *transform_;

  // if accumulate_mode is true, it will, in addition to whatever propagation it is
  // doing, accumulate stats for the model it uses for adaptation, which will be
  // needed in test time since there are no multi-speaker minibatches.
  bool accumulate_mode_;

  // if test_mode_ is true, the forward propagation will assume a single speaker
  // and will use TestingAccumulate() and TestingForward() to do the forward
  // propagation.
  bool test_mode_;


};




} // namespace nnet3
} // namespace kaldi


#endif
