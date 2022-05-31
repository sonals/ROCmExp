# SPDX-License-Identifier: Apache License 2.0 #
# Copyright (c) 2021 Xilinx, Inc. All rights reserved #

SRC = main.cpp
OBJ = main.o
HIPCC = hipcc
CXX = g++
CXXFLAGS = -Wall -D__HIP_PLATFORM_HCC__= -D__HIP_PLATFORM_AMD__ -I/opt/rocm/hip/include -I/opt/rocm/llvm/bin/../lib/clang/14.0.0 -I/opt/rocm/hsa/include
RPROF = rocprof
LDFLAGS = -L/opt/rocm/hip/lib
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
	strace -e trace=ioctl -o strace.log ./a.out
	grep AMDKFD strace.log | awk '-F,' '{print $$2}' | sort | uniq


clean:
	rm -f a.out kernel.co results.* main.o
