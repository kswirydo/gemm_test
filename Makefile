# BACKEND: amd (default) or nvidia
BACKEND ?= amd

CXXFLAGS := -O3 -std=c++17
TARGET   := gemm_test

ifeq ($(BACKEND),nvidia)
    CUDA_PATH ?= /usr/local/cuda
    CXX       := $(CUDA_PATH)/bin/nvcc
    LDFLAGS   := -lcublas
    SRC       := gemm_test_cuda.cpp
else ifeq ($(BACKEND),amd)
    ROCM_PATH      ?= /opt/rocm-7.2.0
    HIPBLASLT_PATH ?= /lustre/orion/ven114/proj-shared/kswirydo/rocm-libraries/install-hipblaslt
    CXX       := $(ROCM_PATH)/bin/hipcc
    CXXFLAGS  += -I$(HIPBLASLT_PATH)/include
    LDFLAGS   := -lrocblas \
                  -L$(HIPBLASLT_PATH)/lib -lhipblaslt \
                  -Wl,--disable-new-dtags \
                  -Wl,-rpath,$(HIPBLASLT_PATH)/lib \
                  -Wl,-rpath,$(ROCM_PATH)/lib/llvm/lib \
                  -Wl,-rpath,$(ROCM_PATH)/lib
    SRC       := gemm_test.cpp
else
    $(error Unknown BACKEND "$(BACKEND)". Use 'amd' or 'nvidia')
endif

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

run: $(TARGET)
	ROCBLAS_USE_HIPBLASLT=1 ./$(TARGET) --sizes test_sizes.txt

clean:
	rm -f $(TARGET)
