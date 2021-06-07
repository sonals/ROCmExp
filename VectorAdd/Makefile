# SPDX-License-Identifier: Apache License 2.0 #
# Copyright (c) 2021 Xilinx, Inc. All rights reserved #

SRC = main.cpp
OBJ = main.o
HIPCC = hipcc
CXX = g++
CXXFLAGS = -Wall -D__HIP_PLATFORM_HCC__= -D__HIP_PLATFORM_AMD__ -I/opt/rocm-4.2.0/hip/include -I/opt/rocm-4.2.0/llvm/bin/../lib/clang/12.0.0 -I/opt/rocm-4.2.0/hsa/include
RPROF = rocprof
LDFLAGS = -L/opt/rocm-4.2.0/hip/lib
LDLIBS = -lamdhip64 -lm -lrt

all: a.out kernel.co

main.o: main.cpp
	$(CXX) -c $(CXXFLAGS) $< -g -o $@

a.out: main.o
	$(CXX) -g $< $(LDFLAGS) $(LDLIBS) -o $@

kernel.co : kernel.cpp
	$(HIPCC) --genco $< -o $@

run: all
	./a.out

profile: all
	$(RPROF) --kfd-trace ./a.out
	$(RPROF) --hsa-trace ./a.out
	jq '.traceEvents[] | .name' results.json | sort | uniq


clean:
	rm -f a.out kernel.co results.* main.o
