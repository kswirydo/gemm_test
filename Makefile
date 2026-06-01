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
    ROCM_PATH ?= /opt/rocm
    CXX       := $(ROCM_PATH)/bin/hipcc
    LDFLAGS   := -lrocblas
    SRC       := gemm_test.cpp
else
    $(error Unknown BACKEND "$(BACKEND)". Use 'amd' or 'nvidia')
endif

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

run: $(TARGET)
	./$(TARGET) test_sizes.txt

clean:
	rm -f $(TARGET)
