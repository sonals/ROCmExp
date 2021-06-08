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
static const int LOOP = 100000;

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

    HIP_ASSERT(hipMemcpy(deviceB.get(), hostB.get(), SIZE, hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(deviceC.get(), hostC.get(), SIZE, hipMemcpyHostToDevice));

    void *tmpA0 = deviceA.get();
    void *tmpB0 = deviceB.get();
    void *tmpC0 = deviceC.get();
    void *args0[] = {&tmpA0, &tmpB0, &tmpC0};

    std::cout << "Run " << KERNELNAME << ' ' << LOOP << " times using device resident memory" << std::endl;
    Timer timer;
    for (int i = 0; i < LOOP; i++) {
        HIP_ASSERT(hipModuleLaunchKernel(function,
                                         LEN/THREADS_PER_BLOCK_X, 1, 1,
                                         THREADS_PER_BLOCK_X, 1, 1,
                                         0, 0, args0, 0));
    }
    HIP_ASSERT(hipDeviceSynchronize());
    auto timer_stop0 = timer.stop();
    std::cout << '(' << LOOP << ", " << timer_stop0 << " ms, " << (LOOP * 1000.0)/timer_stop0
              << " ops/s)" << std::endl;
    HIP_ASSERT(hipMemcpy(hostA.get(), deviceA.get(), SIZE, hipMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < LEN; i++) {
        if (hostA[i] != (hostB[i] + hostC[i])) {
            errors = 1;
            break;
        }
        // This crashes (SIGSEGV) for some reasons
        if (deviceA.get()[i] != (deviceB.get()[i] + deviceC.get()[i])) {
            errors = 1;
            break;
        }
        hostA[i] = 0.0;
        break;
    }

    if (errors)
        std::cout << "FAILED" << std::endl;
    else
        std::cout << "PASSED" << std::endl;

    std::cout << "Run " << KERNELNAME << ' ' << LOOP << " times using host resident memory" << std::endl;
    void *tmpA1 = hostA.get();
    void *tmpB1 = hostB.get();
    void *tmpC1 = hostC.get();
    void *args1[] = {&tmpA1, &tmpB1, &tmpC1};
    timer.reset();
    for (int i = 0; i < LOOP; i++) {
        HIP_ASSERT(hipModuleLaunchKernel(function,
                                         LEN/THREADS_PER_BLOCK_X, 1, 1,
                                         THREADS_PER_BLOCK_X, 1, 1,
                                         0, 0, args1, 0));
    }
    HIP_ASSERT(hipDeviceSynchronize());
    auto timer_stop1 = timer.stop();
    std::cout << '(' << LOOP << ", " << timer_stop1 << " ms, " << (LOOP * 1000.0)/timer_stop1
              << " ops/s)" << std::endl;

    // The following host memory test fails
    for (int i = 0; i < LEN; i++) {
        if (hostA[i] == (hostB[i] + hostC[i]))
            continue;
        errors = 1;
        break;
    }

    if (errors)
        std::cout << "FAILED" << std::endl;
    else
        std::cout << "PASSED" << std::endl;

    return errors;
}
