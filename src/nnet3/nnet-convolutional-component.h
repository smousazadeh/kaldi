// nnet3/nnet-convolutional-component.h

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

#ifndef KALDI_NNET3_NNET_CONVOLUTIONAL_COMPONENT_H_
#define KALDI_NNET3_NNET_CONVOLUTIONAL_COMPONENT_H_

#include "nnet3/nnet-common.h"
#include "nnet3/nnet-component-itf.h"
#include "nnet3/natural-gradient-online.h"
#include "nnet3/convolution.h"
#include <iostream>

namespace kaldi {
namespace nnet3 {

/// @file  nnet-convolutional-component.h
///
/// This file can be viewed as 'overflow' from nnet-general-component.h.
/// It contains a number of components which implement some kind of
/// convolution.


/**
   TimeHeightConvolutionComponent implements 2-dimensional convolution where one
   of the dimensions of convolution (which traditionally would be called the
   width axis) is identified with time (i.e. the 't' component of Indexes).  For
   a deeper understanding of how this works, please see convolution.h.

   The following are the parameters accepted on the config line, with examples
   of their values.


   Parameters inherited from UpdatableComponent (see comment above declaration of
   UpdadableComponent in nnet-component-itf.h for details):
       learning-rate, learning-rate-factor, max-change

   Convolution-related parameters:

     num-filters-in   E.g. num-filters-in=32.  Number of input filters (the
                      number of separate versions of the input image).  The
                      filter-dim has stride 1 in the input and output vectors,
                      i.e. we order the input as (all-filters-for-height=0,
                      all-filters-for-height=1, etc.)
     num-filters-out  E.g. num-filters-out=64. The number of output filters (the
                      number of separate versions of the output image).  As with
                      the input, the filter-dim has stride 1.
     height-in        E.g. height-in=40.  The height of the input image.  The
                      width is not specified the the model level, as it's
                      identified with "t" and is called the time axis; the width
                      is determined by how many "t" values were available at the
                      input of the network, and how many were requested at the
                      output.
     height-out       E.g. height-out=40.  The height of the output image.  Will
                      normally be <= (the input height divided by
                      height-subsample-out).
     height-subsample-out E.g. height-subsample-out=2 (defaults to 1).
                      Subsampling factor on the height axis, e.g. you might set
                      this to 2 if you are doing subsampling on this layer,
                      which would involve discarding every other height
                      increment at the output.  There is no corresponding config
                      for the time dimension, as time subsampling is determined
                      by which 't' values you request at the output, together
                      with the values of 'time-offsets' at different layers of
                      the network.
     height-offsets   E.g. height-offsets=-1,0,1 The set of height offsets that
                      contribute to each output pixel: with the values -1,0,1,
                      height 10 at the output would see data from heights
                      9,10,11 at the input.  These values will normally be
                      consecutive.  Negative values imply zero-padding on the
                      bottom of the image, since output-height 0 is always
                      defined.  Zero-padding at the top of the image is
                      determined in a similar way (e.g. if height-in==height-out
                      and height-offsets=-1,0,1, then there is 1 pixel of
                      padding at the top and bottom of the image).
     time-offsets     E.g. time-offsets=-1,0,1 The time offsets that we require
                      at the input to produce a given output; these are
                      comparable to the offsets used in TDNNs.  Note that the
                      time axis is always numbered using an absolute scheme, so
                      that if there is subsampling on the time axis, then later
                      in the network you'll see time-offsets like "-2,0,2" or
                      "-4,0,4".  Subsampling on the time axis is not explicitly
                      specified but is implicit based on tracking dependencies.
     offsets          Setting 'offsets' is an alternative to setting both
                      height-offsets and time-offsets, that is useful for
                      configurations with less regularity.  It is a semicolon-
                      separated list of pairs (time-offset,height-offset) that
                      might look like: -1,1;-1,0;-1,1;0,1;....;1,1
     required-time-offsets E.g. required-time-offsets=0 (defaults to the same
                      value as time-offsets).  This is a set of time offsets,
                      which if specified must be a nonempty subset of
                      time-offsets; it determines whether zero-padding on the
                      time axis is allowed in cases where there is insufficient
                      input.  If not specified it defaults to the same as
                      'time-offsets', meaning there is no zero-padding on the
                      time axis.  Note: for speech tasks we tend to pad on the
                      time axis with repeats of the first or last frame, rather
                      than zero; and this is handled while preparing the data
                      and not by the core components of the nnet3 framework.  So
                      for speech tasks we wouldn't normally set this value.
     max-memory-mb    Maximum amount of temporary memory, in megabytes, that may
                      be used as temporary matrices in the convolution computation.
                      default=200.0.

   Initialization parameters:
      param-stddev    Standard deviation of the linear parameters of the
                      convolution.  Defaults to sqrt(1.0 / (num-filters-in *
                      num-height-offsets * num-time-offsets)), e.g.
                      sqrt(1.0/(64*3*3)) for a 3x3 kernel with 64 input
                      filters; this value will ensure that the output has
                      unit stddev if the input has unit stddev.
      bias-stddev     Standard deviation of bias terms.  default=0.0.
      init-unit       Defaults to false.  If true, it is required that
                      num-filters-in equal num-filters-out and there should
                      exist a (height, time) offset in the model equal to (0,
                      0).  We will initialize the parameter matrix to be
                      equivalent to the identity transform.  In this case,
                      param-stddev is ignored.


   Natural-gradient related options are below; you won't normally have to
   set these.

      use-natural-gradient e.g. use-natural-gradient=false (defaults to true).
                       You can set this to false to disable the natural gradient
                       updates (you won't normally want to do this).
      rank-out        Rank used in low-rank-plus-unit estimate of the Fisher-matrix
                      factor that has the dimension (num-rows of the parameter
                      space), which equals num-filters-out.  It
                      defaults to the minimum of 80, or half of the number of
                      output filters.
      rank-in         Rank used in low-rank-plus-unit estimate of the Fisher
                      matrix factor which has the dimension (num-cols of the
                      parameter matrix), which has the dimension
                      (num-input-filters * number of time-offsets * number of
                      height-offsets + 1), e.g. num-input-filters * 3 * 3 + 1
                      for a 3x3 kernel (the +1 is for the bias term).
                      It defaults to the minimum of 80, or half the
                      num-rows of the parameter matrix.  [note: I'm considering
                      decreasing this default to e.g. 40 or 20].
      num-minibatches-history
                      This is used setting the 'num_samples_history'
                      configuration value of the natural gradient object.
                      There is no concept of samples (frames) in the
                      application of natural gradient to the convnet, because
                      we do it all on the rows and columns of the derivative.
                      default=4.0.  A larger value means the Fisher matrix is
                      averaged over more minibatches (it's an exponential-decay
                      thing).
      alpha-out       Constant that determines how much we smooth the
                      Fisher-matrix factors with the unit matrix, for the
                      space of dimension num-filters-out.  default=4.0.
      alpha-in        Constant that determines how much we smooth the
                      Fisher-matrix factors with the unit matrix, for the
                      space of dimension (num-filters-in * num-time-offsets *
                      num-height-offsets + 1).  default=4.0.


   Example of a 3x3 kernel with no subsampling, and with zero-padding on both
   the the height and time axis, and where there has previously been no
   subsampling on the time axis:

     num-filters-in=32 num-filters-out=64 height-in=28 height-out=28 \
       height-subsample-out=1 height-offsets=-1,0,1 time-offsets=-1,0,1 \
       required-time-offsets=0

   Example of a 3x3 kernel with no subsampling, without zero-padding on
   either axis, and where there has *previously* been 2-fold subsampling
   on the time axis:

     num-filters-in=32 num-filters-out=64 height-in=20 height-out=18 \
       height-subsample-out=1 height-offsets=0,1,2 time-offsets=0,2,4

   [note: above, the choice to have the time-offsets start at zero rather than
   be centered is just a choice: it assumes that at the output of the network
   you would want to request indexes with t=0, while at the input the t values
   start from zero.]

   Example of a 3x3 kernel with subsampling on the height axis,
   without zero-padding on either axis, and where there has
   previously been 2-fold subsampling on the time axis:

     num-filters-in=32 num-filters-out=64 height-in=20 height-out=9 \
       height-subsample-out=2 height-offsets=0,1,2 time-offsets=0,2,4

  [note: subsampling on the time axis is not expressed in the layer itself:
  any time you increase the distance between time-offsets, like changing
  them from 0,1,2 to 0,2,4, you are effectively subsampling the previous
  layer-- assuming you only request the output at one time value or at
  multiples of the total subsampling factor.]

  Example of a 1x1 kernel:

     num-filters-in=64 num-filters-out=64 height-in=20 height-out=20 \
       height-subsample-out=1 height-offsets=0 time-offsets=0
 */
class TimeHeightConvolutionComponent: public UpdatableComponent {
 public:

  // The use of this constructor should only precede InitFromConfig()
  TimeHeightConvolutionComponent();

  // Copy constructor
  TimeHeightConvolutionComponent(const TimeHeightConvolutionComponent &other);

  virtual int32 InputDim() const;
  virtual int32 OutputDim() const;

  virtual std::string Info() const;
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual std::string Type() const { return "TimeHeightConvolutionComponent"; }
  virtual int32 Properties() const {
    return kUpdatableComponent|kReordersIndexes|kBackpropAdds|
        kBackpropNeedsInput|kInputContiguous|kOutputContiguous;
  }
  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;
  virtual Component* Copy() const {
    return new TimeHeightConvolutionComponent(*this);
  }


  // Some functions that are only to be reimplemented for GeneralComponents.

  // This ReorderIndexes function may insert 'blank' indexes (indexes with
  // t == kNoTime) as well as reordering the indexes.  This is allowed
  // behavior of ReorderIndexes functions.
  virtual void ReorderIndexes(std::vector<Index> *input_indexes,
                              std::vector<Index> *output_indexes) const;

  virtual void GetInputIndexes(const MiscComputationInfo &misc_info,
                               const Index &output_index,
                               std::vector<Index> *desired_indexes) const;

  // This function returns true if at least one of the input indexes used to
  // compute this output index is computable.
  virtual bool IsComputable(const MiscComputationInfo &misc_info,
                            const Index &output_index,
                            const IndexSet &input_index_set,
                            std::vector<Index> *used_inputs) const;

  virtual ComponentPrecomputedIndexes* PrecomputeIndexes(
      const MiscComputationInfo &misc_info,
      const std::vector<Index> &input_indexes,
      const std::vector<Index> &output_indexes,
      bool need_backprop) const;

  // Some functions from base-class UpdatableComponent.
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  virtual void PerturbParams(BaseFloat stddev);
  virtual BaseFloat DotProduct(const UpdatableComponent &other) const;
  virtual int32 NumParameters() const;
  virtual void Vectorize(VectorBase<BaseFloat> *params) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);
  virtual void FreezeNaturalGradient(bool freeze);


  class PrecomputedIndexes: public ComponentPrecomputedIndexes {
   public:
    PrecomputedIndexes() { }
    PrecomputedIndexes(const PrecomputedIndexes &other):
        computation(other.computation) { }
    virtual PrecomputedIndexes *Copy() const;
    virtual void Write(std::ostream &os, bool binary) const;
    virtual void Read(std::istream &os, bool binary);
    virtual std::string Type() const {
      return "TimeHeightConvolutionComponentPrecomputedIndexes";
    }
    virtual ~PrecomputedIndexes() { }

    time_height_convolution::ConvolutionComputation computation;
  };

  void ScaleLinearParams(BaseFloat alpha) { linear_params_.Scale(alpha); }

  void ConsolidateMemory();
 private:

  void Check() const;

  // computes derived parameters required_time_offsets_ and all_time_offsets_.
  void ComputeDerived();

  // Function that updates linear_params_ and bias_params_, which
  // uses the natural gradient code.
  void UpdateNaturalGradient(
      const PrecomputedIndexes &indexes,
      const CuMatrixBase<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv);

  // Function that updates linear_params_ and bias_params_, which
  // does not use the natural gradient code.
  void UpdateSimple(
      const PrecomputedIndexes &indexes,
      const CuMatrixBase<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv);

  // Function called to initialize linear_params_ if init-unit=true in the config
  // line.
  void InitUnit();

  time_height_convolution::ConvolutionModel model_;

  // all_time_offsets_ is a copy of the corresponding variable in
  // model, stored as a vector instead of as a set for efficiency.
  std::vector<int32> all_time_offsets_;
  // time_offset_required_ is a vector with the same dimension as
  // 'all_time_offsets_', which is true if the corresponding time-offset
  // is a member of model_.required_time_offsets_.
  std::vector<bool> time_offset_required_;

  // the linear parameters of the convolution.
  // dimension is model_.ParamRows() by model.ParamCols(),
  // which equals num-filters-out by
  // (num-filters-in * patch-rows * patch-cols),
  // a.k.a.
  // (num-filters-in * num-time-offsets * num-height-offset).
  CuMatrix<BaseFloat> linear_params_;
  // the bias parameters of the convolution, dimension is
  // model_.num_filters_out.
  CuVector<BaseFloat> bias_params_;


  // Maximum amount of temporary memory in megabytes that is allowed to be used
  // in the convolution computation.  (this is per computation, but it's
  // released immediately after it's used, so it doesn't matter how many there
  // are).
  BaseFloat max_memory_mb_;

  // Controls whether or not the natural-gradient is used.
  // Note: even if this is true, if is_gradient_ (from the
  // UpdatableComponent base class) is true, we'll do the 'simple'
  // update that doesn't include natural gradient.
  bool use_natural_gradient_;

  // Preconditioner for the input space, of dimension linear_params_.NumCols() +
  // 1 (the 1 is for the bias).  As with other natural-gradient objects, it's
  // not stored with the model on disk but is reinitialized each time we start
  // up.
  OnlineNaturalGradient preconditioner_in_;

  // Preconditioner for the output space, of dimension
  // linear_params_.NumRows().
  OnlineNaturalGradient preconditioner_out_;
};



/**
   TdnnComponent is a more memory-efficient alternative to manually splicing
   several frames of input and then using a NaturalGradientAffineComponent or
   a LinearComponent.  It does the splicing of the input itself, using
   mechanisms similar to what TimeHeightConvolutionComponent uses.  The
   implementation is in nnet-tdnn-component.cc

   Parameters inherited from UpdatableComponent (see comment above declaration of
   UpdadableComponent in nnet-component-itf.h for details):
       learning-rate, learning-rate-factor, max-change

   Important parameters:

     input-dim         The input feature dimension (before splicing).

     output-dim        The output feature dimension

     time-offsets     E.g. time-offsets=-1,0,1 or time-offsets=-3,0,3.
                      The time offsets that we require at the input to produce a given output.
                      comparable to the offsets used in TDNNs.  They
                      must be unique (no repeats).
     use-bias         Defaults to true, but set to false if you want this to
                      be linear rather than affine in its input.


  Extra parameters:
    orthonormal-constraint=0.0 If you set this to 1.0, then the linear_params_
                      matrix will be (approximately) constrained during training
                      to have orthonormal rows (or columns, whichever is
                      fewer).. it turns out the real name for this is a
                      "semi-orthogonal" matrix.  You can choose a positive
                      nonzero value different than 1.0 to have a scaled
                      semi-orthgonal matrix, i.e. with singular values at the
                      selected value (e.g. 0.5, or 2.0).  This is not enforced
                      inside the component itself; you have to call
                      ConstrainOrthonormal() from the training code to do this.
                      All this component does is return the
                      OrthonormalConstraint() value.  If you set this to a
                      negative value, it's like saying "for any value", i.e. it
                      will constrain the parameter matrix to be closer to "any
                      alpha" times a semi-orthogonal matrix, without changing
                      its overall norm.


   Initialization parameters:
      param-stddev    Standard deviation of the linear parameters of the
                      convolution.  Defaults to
                      sqrt(1.0 / (input-dim * the number of time-offsets))
      bias-stddev     Standard deviation of bias terms.  default=0.0.
                      You should not set this if you set use-bias=false.


   Natural-gradient related options are below; you won't normally have to
   set these as the defaults are reasonable.

      use-natural-gradient e.g. use-natural-gradient=false (defaults to true).
                       You can set this to false to disable the natural gradient
                       updates (you won't normally want to do this).
      rank-out         Rank used in low-rank-plus-unit estimate of the Fisher-matrix
                       factor that has the dimension (num-rows of linear_params_),
                       which equals output_dim.  It
                       defaults to the minimum of 80, or half of the output dim.
      rank-in          Rank used in low-rank-plus-unit estimate of the Fisher
                       matrix factor which has the dimension (num-cols of the
                       parameter matrix), which is input-dim times the number of
                       time offsets.  It defaults to the minimum of 20, or half the
                       num-rows of the parameter matrix.
      num-samples-history
                      This becomes the 'num_samples_history'
                      configuration value of the natural gradient objects.  The
                      default value is 2000.0.

 */
class TdnnComponent: public UpdatableComponent {
 public:

  // The use of this constructor should only precede InitFromConfig()
  TdnnComponent();

  // Copy constructor
  TdnnComponent(const TdnnComponent &other);

  int32 InputDim() const override {
    return linear_params_.NumCols() / static_cast<int32>(time_offsets_.size());
  }
  int32 OutputDim() const override { return linear_params_.NumRows(); }

  std::string Info() const override;
  void InitFromConfig(ConfigLine *cfl) override;
  std::string Type() const override { return "TdnnComponent"; }
  int32 Properties() const override {
    return kUpdatableComponent|kReordersIndexes|kBackpropAdds|
        (bias_params_.Dim() == 0 ? kPropagateAdds : 0)|
        kBackpropNeedsInput;
  }
  void* Propagate(const ComponentPrecomputedIndexes *indexes,
                  const CuMatrixBase<BaseFloat> &in,
                  CuMatrixBase<BaseFloat> *out) const override;
  void Backprop(const std::string &debug_info,
                const ComponentPrecomputedIndexes *indexes,
                const CuMatrixBase<BaseFloat> &in_value,
                const CuMatrixBase<BaseFloat> &out_value,
                const CuMatrixBase<BaseFloat> &out_deriv,
                void *memo,
                Component *to_update,
                CuMatrixBase<BaseFloat> *in_deriv) const override;

  void Read(std::istream &is, bool binary) override;
  void Write(std::ostream &os, bool binary) const override;
  Component* Copy() const override {
    return new TdnnComponent(*this);
  }


  // Some functions that are only to be reimplemented for GeneralComponents.

  // This ReorderIndexes function may insert 'blank' indexes (indexes with
  // t == kNoTime) as well as reordering the indexes.  This is allowed
  // behavior of ReorderIndexes functions.
  void ReorderIndexes(std::vector<Index> *input_indexes,
                      std::vector<Index> *output_indexes) const override;

  void GetInputIndexes(const MiscComputationInfo &misc_info,
                       const Index &output_index,
                       std::vector<Index> *desired_indexes) const override;

  // This function returns true if at least one of the input indexes used to
  // compute this output index is computable.
  bool IsComputable(const MiscComputationInfo &misc_info,
                    const Index &output_index,
                    const IndexSet &input_index_set,
                    std::vector<Index> *used_inputs) const override;

  ComponentPrecomputedIndexes* PrecomputeIndexes(
      const MiscComputationInfo &misc_info,
      const std::vector<Index> &input_indexes,
      const std::vector<Index> &output_indexes,
      bool need_backprop) const override;

  // Some functions from base-class UpdatableComponent.
  void Scale(BaseFloat scale) override;
  void Add(BaseFloat alpha, const Component &other) override;
  BaseFloat DotProduct(const UpdatableComponent &other) const override;
  int32 NumParameters() const override;
  void Vectorize(VectorBase<BaseFloat> *params) const override;
  void UnVectorize(const VectorBase<BaseFloat> &params) override;
  void FreezeNaturalGradient(bool freeze) override;


  class PrecomputedIndexes: public ComponentPrecomputedIndexes {
   public:
    PrecomputedIndexes() { }
    PrecomputedIndexes(const PrecomputedIndexes &other):
        row_stride(other.row_stride), row_offsets(other.row_offsets) { }
    virtual PrecomputedIndexes *Copy() const;
    virtual void Write(std::ostream &os, bool binary) const;
    virtual void Read(std::istream &os, bool binary);
    virtual std::string Type() const {
      return "TdnnComponentPrecomputedIndexes";
    }
    virtual ~PrecomputedIndexes() { }


    // input_row_stride is the stride (in number of rows) we have to take in the
    // input matrix each time we form a sub-matrix that will be part of the
    // input to the tdnn operation.  Normally this will be 1, but it may be,
    // for example, 3 in layers where we do subsampling.
    int32 row_stride;

    // 'row_offsets' is of the same dimension as time_offsets_.  Each element
    // describes the row offset (in the input matrix) of a sub-matrix, and each.
    // We will append together these sub-matrices (row-wise) to be the input to
    // the affine or linear transform.
    std::vector<int32> row_offsets;
  };

  CuMatrixBase<BaseFloat> &LinearParams() { return linear_params_; }

  // This allows you to resize the vector in order to add a bias where
  // there previously was none-- obviously this should be done carefully.
  CuVector<BaseFloat> &BiasParams() { return bias_params_; }

  BaseFloat OrthonormalConstraint() const { return orthonormal_constraint_; }

  void ConsolidateMemory() override;
 protected:

  /**
     This is a piece of the function InitFromConfig() that is broken out and
     made virtual, so that it can be overridden in child-class
     BlockFactorizedTdnnComponent.
     @param [in] num_rows   The number of rows desired in linear_params_, which
                         equals the output-dim of this component
     @param [in] num_cols  The number of columns desired in linear_params, which
                         equals the input-dim of this component multiplied
                         by times_offset_.size() (i.e. the  number of elements
                         in the 'time-offsets' config parameter.
     @param [in,out] cfl  The config line we are parsing.  In the case of
                        TdnnComponent it will only read the 'param-stddev'.
  */
  virtual void InitLinearParams(int32 num_rows,
                                int32 num_cols,
                                ConfigLine *cfl);

  // This static function is a utility function that extracts a CuSubMatrix
  // representing a subset of rows of 'input_matrix'.
  // The numpy syntax would be:
  //   return input_matrix[row_offset:row_stride:num_output_rows*row_stride,:]
  static CuSubMatrix<BaseFloat> GetInputPart(
      const CuMatrixBase<BaseFloat> &input_matrix,
      int32 num_output_rows,
      int32 row_stride,
      int32 row_offset);

  // see the definition for more explanation.
  static void ModifyComputationIo(time_height_convolution::ConvolutionComputationIo *io);

  void Check() const;

  /**
    Function that updates linear_params_, and bias_params_ if present, which
    uses the natural gradient code.
       @param [in] indexes   The indexes that describe the structure of the
                       inputs and outputs.
       @param [in] in_value  The input to this component, e.g. as given to
                      Propagate()
       @param [in] out_value  The derivative of the objective function w.r.t. the
                     output of this component.
  */
  void UpdateNaturalGradient(
      const PrecomputedIndexes &indexes,
      const CuMatrixBase<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv);

  /** Function that updates linear_params_, and bias_params_ if present, which
      does not use the natural gradient code.  The interface is the same as
      UpdateNaturalGradient().  */
  void UpdateSimple(
      const PrecomputedIndexes &indexes,
      const CuMatrixBase<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv);



  // time_offsets_ is the list of time-offsets of the input that
  // we append together; it will typically be (-1,0,1) or (-3,0,3).
  std::vector<int32> time_offsets_;

  // the linear parameters of the network; its NumRows() is the output
  // dim, and its NumCols() equals the input dim times time_offsets_.size().
  // When this object is really child-class BlockFactorizedTdnnComponent
  // we ensure that this is stored with NumCols() == Stride().
  CuMatrix<BaseFloat> linear_params_;

  // the bias parameters if this is an affine transform, or the empty vector if
  // this is a linear operation (i.e. use-bias == false in the config).
  CuVector<BaseFloat> bias_params_;

  // If nonzero, this controls how we apply an orthonormal constraint to the
  // parameter matrix; see docs for ConstrainOrthonormal() in nnet-utils.h.
  // This class just returns the value via the OrthonormalConstraint() function;
  // it doesn't actually do anything with it directly.
  BaseFloat orthonormal_constraint_;

  // Controls whether or not the natural-gradient is used.  Note: even if this
  // is true, if is_gradient_ (from the UpdatableComponent base class) is true,
  // we'll do the 'simple' update that doesn't include natural gradient.
  bool use_natural_gradient_;

  // Preconditioner for the input space, of dimension linear_params_.NumCols() +
  // 1 (the 1 is for the bias).  As with other natural-gradient objects, it's
  // not stored with the model on disk but is reinitialized each time we start
  // up.
  OnlineNaturalGradient preconditioner_in_;

  // Preconditioner for the output space, of dimension
  // linear_params_.NumRows().
  OnlineNaturalGradient preconditioner_out_;
};



/**
   BlockFactorizedTdnnComponent is a modified form of TdnnComponent (which
   inherits from TdnnComponent) that is
   inspired by quaternion-based neural networks, but is more general and trainable--
   the idea is that blocks of parameters are linear functions of
   a smaller number parameters, where the linear function itself is trainable.
   For example, in

         Q_{mat} = [ r  -x  -y  -z
                     x  r   -z  y
                     y  z   r   -x
                     z  -y  x   r  ]

   (where the quaternion is parameterized by r,x,y,z); and we can generalize
   and say the block has four parameters (r,x,y,z) and we have a 16x4 learnable
   matrix that maps from those 4 parameters to the matrix entries.  In
   general there doesn't have to be a correspondence between the number
   of rows and columns of these blocks and the number of parameters per block,
   i.e. the 4 rows of the block, 4 columns of the block and 4 parameters
   don't need to all be the same number.

   Please understand that TdnnComponent is basically an affine component
   that happens to have some advantages in terms of memory efficiency
   when you are doing splicing over time.  You can use it as you would
   use an AffineComponent, and the block factorization doesn't really
   interact with that splicing-over-time stuff (or indeed with the
   bias term). so it might be easier to imagine that we are sub-classing
   a LinearComponent.



   Below we list the config parameters recognized in InitFromConfig().


   The following options are specific to this sub-class, and all are required.

      output-block-dim      The number of rows of each block of linear_params_
                            (must divide output-dim).  Each block, which is a
                            matrix, is obtained as a product of block_basis_
                            with a vector of size params-per-block specific to
                            that block.
      input-block-dim       The number of columns of each block of linear_params_
                            (must divide input-dim).
      params-per-block      The number of real paramters per block (should
                            probably be substantially less than input-block-dim
                            * output-block-dim, or this method doesn't really
                            make sense, but we don't enforce that).

   All the parameters mentioned below are the same as in TdnnComponent.

   Parameters inherited from UpdatableComponent, via TdnnComponent (see comment
   above declaration of UpdadableComponent in nnet-component-itf.h for details):

       learning-rate, learning-rate-factor, max-change

   Parameters inherited from TdnnComponent:

     input-dim         The input feature dimension (before splicing); the num-cols
                       of linear_params_ is this value times time_offsets_.size().

     output-dim        The output feature dimension

     time-offsets     E.g. time-offsets=-1,0,1 or time-offsets=-3,0,3.
                      The time offsets that we require at the input to produce a given output.
                      comparable to the offsets used in TDNNs.  They
                      must be unique (no repeats).
     use-bias         Defaults to true, but set to false if you want this to
                      be linear rather than affine in its input.


     orthonormal-constraint=0.0  You can set this to -1 to enable a 'floating'
                      orthonormal constraint on both parameters
                      reduced_linear_params_ and block_basis_.  Will help
                      stability a lot.  (Values >0 are also possible but not
                      recommended).


   Initialization parameters inherited from TdnnComponent:
      param-stddev    Standard deviation of the linear parameters of the
                      convolution.  Defaults to
                      sqrt(1.0 / (input-dim * the number of time-offsets))
                      TODO: document what is done here.
      bias-stddev     Standard deviation of bias terms.  default=0.0.
                      You should not set this if you set use-bias=false.


   The natural-gradient related options
      use-natural-gradient, rank-out, rank-in, num-samples-history
   are inherited from TdnnComponent; you won't normally have to set these as the
   defaults are reasonable.


 */
class BlockFactorizedTdnnComponent: public TdnnComponent {
 public:

  // The use of this constructor should only precede InitFromConfig()
  BlockFactorizedTdnnComponent() { }

  // Copy constructor
  BlockFactorizedTdnnComponent(const BlockFactorizedTdnnComponent &other);

  // Many class members inherit from TdnnComponent.

  std::string Info() const override;
  std::string Type() const override { return "BlockFactorizedTdnnComponent"; }

  // Propagate() is inherited from class TdnnComponent.
  void Backprop(const std::string &debug_info,
                const ComponentPrecomputedIndexes *indexes,
                const CuMatrixBase<BaseFloat> &in_value,
                const CuMatrixBase<BaseFloat> &out_value,
                const CuMatrixBase<BaseFloat> &out_deriv,
                void *memo,
                Component *to_update,
                CuMatrixBase<BaseFloat> *in_deriv) const override;

  void Read(std::istream &is, bool binary) override;
  void Write(std::ostream &os, bool binary) const override;
  Component* Copy() const override {
    return new BlockFactorizedTdnnComponent(*this);
  }
  void Scale(BaseFloat scale) override;
  void Add(BaseFloat alpha, const Component &other) override;
  BaseFloat DotProduct(const UpdatableComponent &other) const override;
  int32 NumParameters() const override;
  void Vectorize(VectorBase<BaseFloat> *params) const override;
  void UnVectorize(const VectorBase<BaseFloat> &params) override;

  // These members are specific to this class.
  CuMatrixBase<BaseFloat> &ReducedLinearParams() { return reduced_linear_params_; }
  CuMatrixBase<BaseFloat> &BlockParams() { return block_basis_; }

 private:


  void Check() const;

  // This function, called from TdnnComponent::InitFromConfig(),
  // reads the configuration values:
  //    output-block-dim, input-block-dim, params-per-block,
  //    param-stddev
  // See comments in the implementation for further details,
  // e.g. on how param-stddev is interpreted.
  void InitLinearParams(int32 num_rows,
                        int32 num_cols,
                        ConfigLine *cfl) override;


  // This function computes the elements of linear_params_ (the base-class's
  // member object) from the class members reduced_linear_params_ and
  // block_basis_.
  // It requires linear_params_ to already be correctly sized.
  void ComputeLinearParams();

  // Returns the num-cols of each block of parameters in linear_params_.
  inline int32 InputBlockDim() const {
    return linear_params_.NumCols() / NumInputBlocks();
  }
  // Returns the num-rows of each block of parameters in linear_params_.
  inline int32 OutputBlockDim() const {
    return linear_params_.NumRows() / reduced_linear_params_.NumRows();
  }
  // Returns the number of parameters we allocate per block of parameters; would
  // normally be substantially less than InputBlockDim() * OutputBlockDim().
  inline int32 ParamsPerBlock() const { return block_basis_.NumCols(); }
  inline int32 NumInputBlocks() const {
    return reduced_linear_params_.NumCols() / ParamsPerBlock();
  }
  inline int32 NumOutputBlocks() const { return reduced_linear_params_.NumRows(); }

  // This function converts from 'standard' form of the linear transform
  // (i.e. the form linear_params_ is in) to 'intermediate' (reordered) form,
  // which is a matrix of dimension
  //        NumOutputBlocks()  by
  //        NumInputBlocks() * OutputBlockDim() * InputBlockDim()
  // where the 3 factors above are displayed in order of greatest to least stride.
  // This is just a call to CopyCols(), done as its own function for documentation
  // and error checking.
  // We require both linear_params and intermediate_params to have
  // NumCols() == Stride().
  void ConvertToIntermediate(const CuMatrixBase<BaseFloat> &linear_params,
                             CuMatrixBase<BaseFloat> *intermediate_params) const;

  // This does the inverse transformation to ConvertToIntermediate();
  // see its documentation for details on the input and output formats.
  void ConvertToStandard(const CuMatrixBase<BaseFloat> &intermediate_params,
                         CuMatrixBase<BaseFloat> *linear_params) const;


  // This function creates the 'intermediate' form of the parameters from
  // reduced_linear_params_ and block_basis_; intermediate_params
  // should have dimension
  // which is a matrix of dimension
  //        NumOutputBlocks()  by
  //        NumInputBlocks() * OutputBlockDim() * InputBlockDim()
  // where the 3 factors above are displayed in order of greatest to least stride,
  // and should have NumCols() == Stride().  This matrix must be
  // free of NaN's on entry, but does not otherwise have to be defined.
  void MakeIntermediateParams(
      CuMatrixBase<BaseFloat> *intermediate_params) const;


  /**
    Function that updates linear_params_, and bias_params_ if present, which
    uses the natural gradient code (the natural gradient is only used for
    reduced_linear_params_).
                       The component corresponding to the model parameters
                       ('this' corresponds to the derivative_.
       @param [in] indexes   The indexes that describe the structure of the
                       inputs and outputs.
       @param [in] in_value  The input to this component, e.g. as given to
                      Propagate()
       @param [in] out_value  The derivative of the objective function w.r.t. the
                     output of this component.
  */
  void UpdateNaturalGradient(
      const BlockFactorizedTdnnComponent &model_component,
      const PrecomputedIndexes &indexes,
      const CuMatrixBase<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv);

  /** Function that updates linear_params_, and bias_params_ if present, which
      does not use the natural gradient code.  The interface is the same as
      UpdateNaturalGradient().  */
  void UpdateSimple(
      const BlockFactorizedTdnnComponent &model_component,
      const PrecomputedIndexes &indexes,
      const CuMatrixBase<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv);

  // Sets up the arrays to_final_indexes_ and to_intermediate_indexes_.
  void CreateIndexes();

  // Used in ConvertToStandard(), set up by CreateIndexes();
  CuArray<int32> to_standard_indexes_;

  // Used in ConvertToIntermediate(), set up by CreateIndexes().
  CuArray<int32> to_intermediate_indexes_;


  // reduced_linear_params_ and block_basis_ are the 'real' parameters
  // underlying linear_params_.

  // reduced_linear_params_ is of dimension NumOutputBlocks() by
  // (NumInputBlocks() * ParamsPerBlock()), where in the column indexes,
  // the input block number has a higher stride than the parameter within the
  // block.
  // This matrix is stored contiguously (i.e., kStrideEqualNumCols),
  // which is necessary given some striding/reshaping that we do
  // in the implementation.
  CuMatrix<BaseFloat> reduced_linear_params_;

  // block_basis_ is of dimension
  //    (OutputBlockDim() * InputBlockDim())  by  ParamsPerBlock().
  // It contains the mapping from compressed vector form of each block,
  // to full matrix form.
  // The InputBlockDim() and OutputBlockDim() are in order of larger
  // to smaller to stride (i.e. the input blocks are on consecutive rows).
  CuMatrix<BaseFloat> block_basis_;
};





} // namespace nnet3
} // namespace kaldi


#endif
