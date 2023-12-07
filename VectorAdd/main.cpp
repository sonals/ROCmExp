/* SPDX-License-Identifier: Apache License 2.0 */
/* Copyright (c) 2021-2022 Xilinx, Inc. All rights reserved */
/* Copyright (C) 2022-2023 Advanced Micro Devices, Inc. */
/* Based on https://github.com/ROCm-Developer-Tools/HIP-Examples/tree/master/vectorAdd */

#include <cstring>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include "hip/hip_runtime_api.h"

#include "hiperror.h"

#define FILENAME "kernel.co"
#define KERNELNAME "vectoradd"

#define NOP_FILENAME "nop.co"
#define NOP_KERNELNAME "mynop"

static const int LEN = 0x100000;
static const int SIZE = LEN * sizeof(float);
static const int THREADS_PER_BLOCK_X = 32;
static const int LOOP = 5000;

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
        errorCheck(hipMalloc((void**)&_buffer, size * sizeof(T)));
    }
    ~deviceBO() noexcept {
        errorCheck(hipFree(_buffer));
    }
    T *get() const {
        return _buffer;
    }

    T *&get() {
        return _buffer;
    }

};

void runkernel(hipFunction_t function, void *args[])
{
    const char *name = hipKernelNameRef(function);
    std::cout << "Running " << name << ' ' << LOOP << " times...\n";
    Timer timer;

    const int globalr = std::strcmp(name, NOP_KERNELNAME) ? LEN/THREADS_PER_BLOCK_X : 1;
    const int localr = std::strcmp(name, NOP_KERNELNAME) ? THREADS_PER_BLOCK_X : 1;

    for (int i = 0; i < LOOP; i++) {
        errorCheck(hipModuleLaunchKernel(function,
                                         globalr, 1, 1,
                                         localr, 1, 1,
                                         0, 0, args, nullptr), name);
    }
    errorCheck(hipDeviceSynchronize());
    auto delayD = timer.stop();

    std::cout << "Throughput metrics" << std::endl;
    std::cout << '(' << LOOP << " loops, " << delayD << " us, " << (LOOP * 1000000.0)/delayD
              << " ops/s, " << delayD/LOOP << " us average pipelined latency)" << std::endl;


    timer.reset();
    for (int i = 0; i < LOOP; i++) {
        errorCheck(hipModuleLaunchKernel(function,
                                         globalr, 1, 1,
                                         localr, 1, 1,
                                         0, 0, args, nullptr), name);
        errorCheck(hipDeviceSynchronize());
    }

    delayD = timer.stop();

    std::cout << "Latency metrics" << std::endl;
    std::cout << '(' << LOOP << " loops, " << delayD << " us, " << (LOOP * 1000000.0)/delayD
              << " ops/s, " << delayD/LOOP << " us average start-to-finish latency)" << std::endl;

}

int mainworker() {
    hipDeviceProp_t devProp;
    errorCheck(hipGetDeviceProperties(&devProp, 0));

    hipModule_t module;
    hipFunction_t function;
    errorCheck(hipModuleLoad(&module, FILENAME), FILENAME);
    errorCheck(hipModuleGetFunction(&function, module, KERNELNAME), KERNELNAME);

    hipModule_t nopmodule;
    hipFunction_t nopfunction;
    errorCheck(hipModuleLoad(&nopmodule, NOP_FILENAME), NOP_FILENAME);
    errorCheck(hipModuleGetFunction(&nopfunction, nopmodule, NOP_KERNELNAME), NOP_KERNELNAME);


    std::cout << devProp.name << std::endl;
    std::cout << devProp.totalGlobalMem/0x100000 << " MB" << std::endl;
    std::cout << devProp.maxThreadsPerBlock << " Threads" << std::endl;

    std::unique_ptr<float[]> hostA(new float[LEN]);
    std::unique_ptr<float[]> hostB(new float[LEN]);
    std::unique_ptr<float[]> hostC(new float[LEN]);

    // Initialize input/output vectors
    for (int i = 0; i < LEN; i++) {
        hostB[i] = i;
        hostC[i] = i * 2;
        hostA[i] = 0;
    }

    deviceBO<float> deviceA(LEN);
    deviceBO<float> deviceB(LEN);
    deviceBO<float> deviceC(LEN);

    // Sync host buffers to device
    errorCheck(hipMemcpy(deviceB.get(), hostB.get(), SIZE, hipMemcpyHostToDevice));
    errorCheck(hipMemcpy(deviceC.get(), hostC.get(), SIZE, hipMemcpyHostToDevice));

    void *argsD[] = {&deviceA.get(), &deviceB.get(), &deviceC.get()};

    std::cout << "---------------------------------------------------------------------------------\n";
    std::cout << "Run " << hipKernelNameRef(function) << ' ' << LOOP << " times using device resident memory" << std::endl;
    std::cout << "Host buffers: " << hostA.get() << ", "
              << hostB.get() << ", " << hostC.get() << std::endl;
    std::cout << "Device buffers: " << deviceA.get() << ", "
              << deviceB.get() << ", " << deviceC.get() << std::endl;

    runkernel(function, argsD);
    // Sync device output buffer to host
    errorCheck(hipMemcpy(hostA.get(), deviceA.get(), SIZE, hipMemcpyDeviceToHost));

    // Verify output and then reset it for the subsequent test
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

    std::cout << "---------------------------------------------------------------------------------\n";

    std::cout << "Run " << hipKernelNameRef(nopfunction) << ' ' << LOOP << " times using device resident memory" << std::endl;
    std::cout << "Host buffers: " << hostA.get() << ", "
              << hostB.get() << ", " << hostC.get() << std::endl;
    std::cout << "Device buffers: " << deviceA.get() << ", "
              << deviceB.get() << ", " << deviceC.get() << std::endl;

    runkernel(nopfunction, argsD);

    // Register our buffer with ROCm so it is pinned and prepare for access by device
    errorCheck(hipHostRegister(hostA.get(), SIZE, hipHostRegisterDefault));
    errorCheck(hipHostRegister(hostB.get(), SIZE, hipHostRegisterDefault));
    errorCheck(hipHostRegister(hostC.get(), SIZE, hipHostRegisterDefault));

    void *tmpA1 = nullptr;
    void *tmpB1 = nullptr;
    void *tmpC1 = nullptr;

    // Map the host buffer to device address space so device can access the buffers
    errorCheck(hipHostGetDevicePointer(&tmpA1, hostA.get(), 0));
    errorCheck(hipHostGetDevicePointer(&tmpB1, hostB.get(), 0));
    errorCheck(hipHostGetDevicePointer(&tmpC1, hostC.get(), 0));

    std::cout << "---------------------------------------------------------------------------------\n";
    std::cout << "Run " << hipKernelNameRef(function) << ' ' << LOOP << " times using host resident memory" << std::endl;
    std::cout << "Device mapped host buffers: " << tmpA1 << ", "
              << tmpB1 << ", " << tmpC1 << std::endl;

    void *argsH[] = {&tmpA1, &tmpB1, &tmpC1};

    runkernel(function, argsH);
    // Verify the output
    for (int i = 0; i < LEN; i++) {
        if (hostA[i] == (hostB[i] + hostC[i]))
            continue;
        errors++;
        break;
    }

    std::cout << "---------------------------------------------------------------------------------\n";
    std::cout << "Run " << hipKernelNameRef(nopfunction) << ' ' << LOOP << " times using host resident memory" << std::endl;
    std::cout << "Device mapped host buffers: " << tmpA1 << ", "
              << tmpB1 << ", " << tmpC1 << std::endl;

    runkernel(nopfunction, argsH);

    // Unmap the host buffers from device address space
    errorCheck(hipHostUnregister(hostC.get()));
    errorCheck(hipHostUnregister(hostB.get()));
    errorCheck(hipHostUnregister(hostA.get()));

    if (errors)
        std::cout << "FAILED" << std::endl;
    else
        std::cout << "PASSED" << std::endl;


    return errors;
}

int main()
{
    try {
        int errors = mainworker();

    } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}
