# SPDX-License-Identifier: Apache License 2.0 #
# Copyright (c) 2021 Xilinx, Inc. All rights reserved #

ROCM_ROOT = /opt/rocm
SRC = main.cpp
OBJ = main.o
HIPCC = $(ROCM_ROOT)/bin/hipcc
CXX = g++
CC = g++
CXXFLAGS = -Wall -D__HIP_PLATFORM_HCC__= -D__HIP_PLATFORM_AMD__ -I$(ROCM_ROOT)/include -I$(ROCM_ROOT)/llvm/bin/../lib/clang/14.0.0 -I$(ROCM_ROOT)/hsa/include
RPROF = $(ROCM_ROOT)/rocprof
LDFLAGS = -L$(ROCM_ROOT)/hip/lib
LDLIBS = -lamdhip64 -lm -lrt

debug ?= 0
ifeq ($(debug), 1)
    CXXFLAGS +=-DDEBUG -g
else
    CXXFLAGS +=-DNDEBUG -O2
endif

all: main kernel.co

main: main.o
#	$(CXX) $< $(LDFLAGS) $(LDLIBS) -o $@

kernel.co : kernel.cpp
	$(HIPCC) --genco $< -o $@

run: all
	./main

profile: all
	$(RPROF) --hip-trace ./main
	$(RPROF) --hsa-trace ./main
	jq '.traceEvents[] | .name' results.json | sort | uniq
	strace -e trace=ioctl -o strace.log ./main
	grep AMDKFD strace.log | awk '-F,' '{print $$2}' | sort | uniq

compile_commands.json: main.cpp kernel.cpp
	bear -- make debug=1 all

compdb: compile_commands.json

clean:
	rm -f main kernel.co results.* main.o
