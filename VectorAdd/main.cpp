/* SPDX-License-Identifier: Apache License 2.0 */
/* Copyright (c) 2021 Xilinx, Inc. All rights reserved */
/* Based on https://github.com/ROCm-Developer-Tools/HIP-Examples/tree/master/vectorAdd */

#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <chrono>
#include <stdlib.h>
#include <iostream>
#include <memory>
#include "hip/hip_runtime_api.h"


#define HIP_ASSERT(x) (assert((x)==hipSuccess))
#define HIP_CHECK(status)                                                                          \
    if (status != hipSuccess) {                                                                    \
        std::cout << "Got Status: " << status << " at Line: " << __LINE__ << std::endl;            \
        exit(0);                                                                                   \
    }

#define FILENAME "kernel.co"
#define KERNELNAME "vectoradd"

static const int LEN = 0x100000;
static const int SIZE = LEN * sizeof(float);
static const int THREADS_PER_BLOCK_X = 32;
static const int LOOP = 1000;

class Timer {
    std::chrono::high_resolution_clock::time_point mTimeStart;
public:
    Timer() {
        reset();
    }
    long long stop() {
        std::chrono::high_resolution_clock::time_point timeEnd = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - mTimeStart).count();
    }
    void reset() {
        mTimeStart = std::chrono::high_resolution_clock::now();
    }
};

// Abstraction of device buffer so we can do automatic buffer dealocation (RAII)
template<typename T> class deviceBO {
    T *_buffer;
public:
    deviceBO(size_t size) : _buffer(nullptr) {
        HIP_ASSERT(hipMalloc((void**)&_buffer, size * sizeof(T)));
    }
    ~deviceBO() {
        HIP_ASSERT(hipFree(_buffer));
    }
    T *get() const {
        return _buffer;
    }

    T *&get() {
        return _buffer;
    }

};

int main() {
    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);

    hipModule_t module;
    hipFunction_t function;
    HIP_ASSERT(hipModuleLoad(&module, FILENAME));
    HIP_ASSERT(hipModuleGetFunction(&function, module, KERNELNAME));

    std::cout << devProp.name << std::endl;
    std::cout << devProp.totalGlobalMem/0x100000 << " MB" << std::endl;
    std::cout << devProp.maxThreadsPerBlock << " Threads" << std::endl;

    std::unique_ptr<float[]> hostA(new float[LEN]);
    std::unique_ptr<float[]> hostB(new float[LEN]);
    std::unique_ptr<float[]> hostC(new float[LEN]);

    for (int i = 0; i < LEN; i++) {
        hostB[i] = i;
        hostC[i] = i * 2;
        hostA[i] = 0;
    }

    deviceBO<float> deviceA(LEN);
    deviceBO<float> deviceB(LEN);
    deviceBO<float> deviceC(LEN);

    // Sync host buffers to device
    HIP_ASSERT(hipMemcpy(deviceB.get(), hostB.get(), SIZE, hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(deviceC.get(), hostC.get(), SIZE, hipMemcpyHostToDevice));

    void *argsD[] = {&deviceA.get(), &deviceB.get(), &deviceC.get()};

    std::cout << "Run " << KERNELNAME << ' ' << LOOP << " times using device resident memory" << std::endl;
    std::cout << "Host buffers: " << hostA.get() << ", "
              << hostB.get() << ", " << hostC.get() << std::endl;
    std::cout << "Device buffers: " << deviceA.get() << ", "
              << deviceB.get() << ", " << deviceC.get() << std::endl;

    Timer timer;
    for (int i = 0; i < LOOP; i++) {
        HIP_ASSERT(hipModuleLaunchKernel(function,
                                         LEN/THREADS_PER_BLOCK_X, 1, 1,
                                         THREADS_PER_BLOCK_X, 1, 1,
                                         0, 0, argsD, nullptr));
    }
    HIP_ASSERT(hipDeviceSynchronize());
    auto delayD = timer.stop();
    std::cout << '(' << LOOP << " loops, " << delayD << " ms, " << (LOOP * 1000.0)/delayD
              << " ops/s)" << std::endl;

    // Sync device output buffer to host
    HIP_ASSERT(hipMemcpy(hostA.get(), deviceA.get(), SIZE, hipMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < LEN; i++) {
        if (hostA[i] != (hostB[i] + hostC[i])) {
            errors++;
            break;
        }
        hostA[i] = 0.0;
    }

    if (errors)
        std::cout << "FAILED" << std::endl;
    else
        std::cout << "PASSED" << std::endl;

    // Register our buffer wit ROCm so it is pinned and prepare for access by device
    HIP_ASSERT(hipHostRegister(hostA.get(), SIZE, hipHostRegisterDefault));
    HIP_ASSERT(hipHostRegister(hostB.get(), SIZE, hipHostRegisterDefault));
    HIP_ASSERT(hipHostRegister(hostC.get(), SIZE, hipHostRegisterDefault));

    void *tmpA1 = nullptr;
    void *tmpB1 = nullptr;
    void *tmpC1 = nullptr;

    // Map the host buffer to device address space
    HIP_ASSERT(hipHostGetDevicePointer(&tmpA1, hostA.get(), 0));
    HIP_ASSERT(hipHostGetDevicePointer(&tmpB1, hostB.get(), 0));
    HIP_ASSERT(hipHostGetDevicePointer(&tmpC1, hostC.get(), 0));

    std::cout << "Run " << KERNELNAME << ' ' << LOOP << " times using host resident memory" << std::endl;
    std::cout << "Device mapped host buffers: " << tmpA1 << ", "
              << tmpB1 << ", " << tmpC1 << std::endl;

    void *argsH[] = {&tmpA1, &tmpB1, &tmpC1};
    timer.reset();
    for (int i = 0; i < LOOP; i++) {
        HIP_ASSERT(hipModuleLaunchKernel(function,
                                         LEN/THREADS_PER_BLOCK_X, 1, 1,
                                         THREADS_PER_BLOCK_X, 1, 1,
                                         0, 0, argsH, nullptr));
    }
    HIP_ASSERT(hipDeviceSynchronize());
    auto delayH = timer.stop();
    std::cout << '(' << LOOP << " loops, " << delayH << " ms, " << (LOOP * 1000.0)/delayH
              << " ops/s)" << std::endl;

    // The following host memory test fails
    for (int i = 0; i < LEN; i++) {
        if (hostA[i] == (hostB[i] + hostC[i]))
            continue;
        errors = 1;
        break;
    }

    // Unmap the host buffers from device address space
    HIP_ASSERT(hipHostUnregister(hostC.get()));
    HIP_ASSERT(hipHostUnregister(hostB.get()));
    HIP_ASSERT(hipHostUnregister(hostA.get()));

    if (errors)
        std::cout << "FAILED" << std::endl;
    else
        std::cout << "PASSED" << std::endl;

    return errors;
}
