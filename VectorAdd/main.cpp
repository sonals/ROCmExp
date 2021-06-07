/* SPDX-License-Identifier: Apache License 2.0 */
/* Copyright (c) 2021 Xilinx, Inc. All rights reserved */
/* Based on https://github.com/ROCm-Developer-Tools/HIP-Examples/tree/master/vectorAdd */

#include <assert.h>
#include <stdio.h>
#include <algorithm>
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

#define fileName "kernel.co"
#define kernel_name "vectoradd"

static const int LEN = 0x100000;
static const int SIZE = LEN * sizeof(float);
static const int THREADS_PER_BLOCK_X = 32;

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
    HIP_ASSERT(hipModuleLoad(&module, fileName));
    HIP_ASSERT(hipModuleGetFunction(&function, module, kernel_name));

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

    deviceBO<float> deviceA(SIZE);
    deviceBO<float> deviceB(SIZE);
    deviceBO<float> deviceC(SIZE);

    HIP_ASSERT(hipMemcpy(deviceB.get(), hostB.get(), SIZE, hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(deviceC.get(), hostC.get(), SIZE, hipMemcpyHostToDevice));

    void *tmpA = deviceA.get();
    void *tmpB = deviceB.get();
    void *tmpC = deviceC.get();
    void *args[] = {&tmpA, &tmpB, &tmpC};
    HIP_ASSERT(hipModuleLaunchKernel(function,
                                     LEN/THREADS_PER_BLOCK_X, 1, 1,
                                     THREADS_PER_BLOCK_X, 1, 1,
                                     0, 0, args, 0));

    HIP_ASSERT(hipMemcpy(hostA.get(), deviceA.get(), SIZE, hipMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < LEN; i++) {
        if (hostA[i] == (hostB[i] + hostC[i]))
            continue;
        if (deviceA.get()[i] == (deviceB.get()[i] + deviceC.get()[i]))
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
