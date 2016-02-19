
all:

OPENFST_CXXFLAGS = 
OPENFST_LDLIBS = 
include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

TESTFILES = ivector-extractor-test plda-test logistic-regression-test xvector-test

OBJFILES = ivector-extractor.o voice-activity-detection.o plda.o logistic-regression.o xvector.o \
           nnet-xvector-training.o nnet-xvector-diagnostics.o 

LIBNAME = kaldi-ivector

ADDLIBS = ../gmm/kaldi-gmm.a ../tree/kaldi-tree.a ../transform/kaldi-transform.a \
          ../thread/kaldi-thread.a ../nnet3/kaldi-nnet3.a ../cudamatrix/kaldi-cudamatrix.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a \
          ../util/kaldi-util.a

include ../makefiles/default_rules.mk
