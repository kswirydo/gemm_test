ROCM_PATH ?= /opt/rocm
HIPCC      := $(ROCM_PATH)/bin/hipcc

CXXFLAGS := -O3 -std=c++17
LDFLAGS  := -lrocblas

TARGET := gemm_test
SRC    := gemm_test.cpp

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCC) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

run: $(TARGET)
	./$(TARGET) test_sizes.txt

clean:
	rm -f $(TARGET)
