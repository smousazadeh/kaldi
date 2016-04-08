// feat/feature-fbank.cc

// Copyright 2009-2012  Karel Vesely
//                2016  Johns Hopkins University (author: Daniel Povey)

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


#include "feat/feature-fbank.h"


namespace kaldi {

FbankComputer::FbankComputer(const FbankOptions &opts):
    opts_(opts), srfft_(NULL) {
  if (opts.energy_floor > 0.0)
    log_energy_floor_ = Log(opts.energy_floor);

  int32 padded_window_size = opts.frame_opts.PaddedWindowSize();
  if ((padded_window_size & (padded_window_size-1)) == 0)  // Is a power of two...
    srfft_ = new SplitRadixRealFft<BaseFloat>(padded_window_size);

  // We'll definitely need the filterbanks info for VTLN warping factor 1.0.
  // [note: this call caches it.]
  GetMelBanks(1.0);
}

FbankComputer::FbankComputer(const FbankComputer &other):
    opts_(other.opts), log_energy_floor_(other.log_energy_floor_),
    mel_banks_(other.mel_banks_), srfft_(NULL) {
  for (std::map<BaseFloat, MelBanks*>::iterator iter = mel_banks_.begin();
      iter != mel_banks_.end();
      ++iter)
    iter->second = new MelBanks(*(iter->second));
  if (other.srfft_)
    srfft_ = new SplitRadixRealFft<BaseFloat>(*(other->srfft_));
}

Fbank::~Fbank() {
  for (std::map<BaseFloat, MelBanks*>::iterator iter = mel_banks_.begin();
      iter != mel_banks_.end(); ++iter)
    delete iter->second;
  delete srfft_;
}

const MelBanks *Fbank::GetMelBanks(BaseFloat vtln_warp) {
  MelBanks *this_mel_banks = NULL;
  std::map<BaseFloat, MelBanks*>::iterator iter = mel_banks_.find(vtln_warp);
  if (iter == mel_banks_.end()) {
    this_mel_banks = new MelBanks(opts_.mel_opts,
                                  opts_.frame_opts,
                                  vtln_warp);
    mel_banks_[vtln_warp] = this_mel_banks;
  } else {
    this_mel_banks = iter->second;
  }
  return this_mel_banks;
}

void Fbank::Compute(const VectorBase<BaseFloat> &wave,
                    BaseFloat vtln_warp,
                    Matrix<BaseFloat> *output,
                    Vector<BaseFloat> *wave_remainder) {
  ComputeInternal(wave, *this_mel_banks, output, wave_remainder);
}

void FbankComputer::Compute(BaseFloat signal_log_energy,
                            BaseFloat vtln_warp,
                            VectorBase<BaseFloat> *signal_frame,
                            VectorBase<BaseFloat> *feature) {

  const MelBanks &mel_banks = *(GetMelBanks(vtln_warp));

  KALDI_ASSERT(signal_frame->Dim() == opts_.frame_opts.PaddedWindowSize() &&
               feature->Dim() == this->Dim());

  // Buffers
  Vector<BaseFloat> window;  // windowed waveform.
  Vector<BaseFloat> mel_energies;


  // Compute energy after window function (not the raw one)
  if (opts_.use_energy && !opts_.raw_energy)
    signal_log_energy = Log(std::max(VecVec(window, window),
                                     std::numeric_limits<BaseFloat>::min()));

  if (srfft_ != NULL)  // Compute FFT using split-radix algorithm.
    srfft_->Compute(signal_frame->Data(), true);
  else  // An alternative algorithm that works for non-powers-of-two.
    RealFft(&window, true);

    // Convert the FFT into a power spectrum.
    ComputePowerSpectrum(&window);
    SubVector<BaseFloat> power_spectrum(window, 0, window.Dim()/2 + 1);

    // Sum with MelFiterbank over power spectrum
    mel_banks.Compute(power_spectrum, &mel_energies);
    if (opts_.use_log_fbank) {
      // avoid log of zero (which should be prevented anyway by dithering).
      mel_energies.ApplyFloor(std::numeric_limits<BaseFloat>::min());
      mel_energies.ApplyLog();  // take the log.
    }

    // Output buffers
    SubVector<BaseFloat> this_output(output->Row(r));
    SubVector<BaseFloat> this_fbank(this_output.Range((opts_.use_energy? 1 : 0),
                                                      opts_.mel_opts.num_bins));

    // Copy to output
    this_fbank.CopyFromVec(mel_energies);
    // Copy energy as first value
    if (opts_.use_energy) {
      if (opts_.energy_floor > 0.0 && log_energy < log_energy_floor_) {
        log_energy = log_energy_floor_;
      }
      this_output(0) = log_energy;
    }

    // HTK compat: Shift features, so energy is last value
    if (opts_.htk_compat && opts_.use_energy) {
      BaseFloat energy = this_output(0);
      for (int32 i = 0; i < opts_.mel_opts.num_bins; i++) {
        this_output(i) = this_output(i+1);
      }
      this_output(opts_.mel_opts.num_bins) = energy;
    }
  }
}

}  // namespace kaldi
