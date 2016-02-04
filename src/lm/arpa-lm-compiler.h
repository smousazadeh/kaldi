
#ifndef KALDI_LM_ARPA_LM_COMPILER_H_
#define KALDI_LM_ARPA_LM_COMPILER_H_

#include <algorithm>
#include <limits>
#include <sstream>
#include <utility>

#include "base/kaldi-math.h"
#include "lm/arpa-file-parser.h"
#include "lm/const-arpa-lm.h"
#include "util/stl-utils.h"
#include "util/text-utils.h"

namespace kaldi {

class ArpaLmCompilerImplInterface;

class ArpaLmCompiler : public ArpaFileParser {
 public:
  ArpaLmCompiler(ArpaParseOptions options, int sub_eps,
                 fst::SymbolTable* symbols)
      : ArpaFileParser(options, symbols),
        sub_eps_(sub_eps), impl_(NULL) {
  }
  ~ArpaLmCompiler();

  const fst::StdVectorFst& Fst() const { return fst_; }

 protected:
  // ArpaFileParser overrides.
  virtual void HeaderAvailable();
  virtual void ConsumeNGram(const NGram& ngram);
  virtual void ReadComplete();

 private:
  int sub_eps_;
  ArpaLmCompilerImplInterface* impl_;  // Owned.
  fst::StdVectorFst fst_;
};

}  // namespace kaldi

#endif  // KALDI_LM_ARPA_LM_COMPILER_H_
