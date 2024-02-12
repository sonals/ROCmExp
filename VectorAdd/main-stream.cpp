/* SPDX-License-Identifier: Apache License 2.0 */
/* Copyright (c) 2021-2022 Xilinx, Inc. All rights reserved */
/* Copyright (C) 2022-2023 Advanced Micro Devices, Inc. */
/* Based on https://github.com/ROCm-Developer-Tools/HIP-Examples/tree/master/vectorAdd */

#include <cstring>
#include <algorithm>
#include <iostream>
#include <memory>
#include <thread>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>

#include "hip/hip_runtime_api.h"

#include "common.h"

#define FILENAME "kernel.co"
#define KERNELNAME "vectoradd"

#define NOP_FILENAME "nop.co"
#define NOP_KERNELNAME "mynop"

namespace {

static const int LEN = 0x100000;
static const int SIZE = LEN * sizeof(float);
static const int THREADS_PER_BLOCK_X = 32;
static const int LOOP = 1000;


void runkernel(hipFunction_t function, hipStream_t stream, void *args[])
{
    const char *name = hipKernelNameRef(function);
    std::cout << "Running " << name << ' ' << LOOP << " times...\n";
    Timer timer;

    const int globalr = std::strcmp(name, NOP_KERNELNAME) ? LEN/THREADS_PER_BLOCK_X : 1;
    const int localr = std::strcmp(name, NOP_KERNELNAME) ? THREADS_PER_BLOCK_X : 1;

    for (int i = 0; i < LOOP; i++) {
        hipCheck(hipModuleLaunchKernel(function,
                                         globalr, 1, 1,
                                         localr, 1, 1,
                                         0, stream, args, nullptr), name);
    }
    hipCheck(hipStreamSynchronize(stream));
    auto delayD = timer.stop();

    std::cout << "Throughput metrics" << std::endl;
    std::cout << '(' << LOOP << " loops, " << delayD << " us, " << (LOOP * 1000000.0)/delayD
              << " ops/s, " << delayD/LOOP << " us average pipelined latency)" << std::endl;

}

int mainworkerthread(hipFunction_t function, hipStream_t stream, bool validate = true) {

    std::cout << "*********************************************************************************\n";

    std::unique_ptr<float[]> hostA(new float[LEN]);
    std::unique_ptr<float[]> hostB(new float[LEN]);
    std::unique_ptr<float[]> hostC(new float[LEN]);

    // Initialize input/output vectors
    for (int i = 0; i < LEN; i++) {
        hostB[i] = i;
        hostC[i] = i * 2;
        hostA[i] = 0;
    }

    DeviceBO<float> deviceA(LEN);
    DeviceBO<float> deviceB(LEN);
    DeviceBO<float> deviceC(LEN);

    // Sync host buffers to device
    hipCheck(hipMemcpyWithStream(deviceB.get(), hostB.get(), SIZE, hipMemcpyHostToDevice, stream));
    hipCheck(hipMemcpyWithStream(deviceC.get(), hostC.get(), SIZE, hipMemcpyHostToDevice, stream));

    void *argsD[] = {&deviceA.get(), &deviceB.get(), &deviceC.get()};

    std::cout << "---------------------------------------------------------------------------------\n";
    std::cout << "Run " << hipKernelNameRef(function) << ' ' << LOOP << " times using device resident memory" << std::endl;
    std::cout << "Host buffers: " << hostA.get() << ", "
              << hostB.get() << ", " << hostC.get() << std::endl;
    std::cout << "Device buffers: " << deviceA.get() << ", "
              << deviceB.get() << ", " << deviceC.get() << std::endl;

    runkernel(function, stream, argsD);
    // Sync device output buffer to host
    hipCheck(hipMemcpyWithStream(hostA.get(), deviceA.get(), SIZE, hipMemcpyDeviceToHost, stream));

    // Verify output and then reset it for the subsequent test
    int errors = 0;
    if (validate) {
        for (int i = 0; i < LEN; i++) {
            if (hostA[i] != (hostB[i] + hostC[i])) {
                errors++;
                break;
            }
            hostA[i] = 0.0;
        }
    }

    if (errors)
        std::cout << "FAILED" << std::endl;
    else
        std::cout << "PASSED" << std::endl;

    // Register our buffer with ROCm so it is pinned and prepare for access by device
    hipCheck(hipHostRegister(hostA.get(), SIZE, hipHostRegisterDefault));
    hipCheck(hipHostRegister(hostB.get(), SIZE, hipHostRegisterDefault));
    hipCheck(hipHostRegister(hostC.get(), SIZE, hipHostRegisterDefault));

    void *tmpA1 = nullptr;
    void *tmpB1 = nullptr;
    void *tmpC1 = nullptr;

    // Map the host buffer to device address space so device can access the buffers
    hipCheck(hipHostGetDevicePointer(&tmpA1, hostA.get(), 0));
    hipCheck(hipHostGetDevicePointer(&tmpB1, hostB.get(), 0));
    hipCheck(hipHostGetDevicePointer(&tmpC1, hostC.get(), 0));

    std::cout << "---------------------------------------------------------------------------------\n";
    std::cout << "Run " << hipKernelNameRef(function) << ' ' << LOOP << " times using host resident memory" << std::endl;
    std::cout << "Device mapped host buffers: " << tmpA1 << ", "
              << tmpB1 << ", " << tmpC1 << std::endl;

    void *argsH[] = {&tmpA1, &tmpB1, &tmpC1};

    runkernel(function, stream, argsH);

    if (validate) {
        // Verify the output
        for (int i = 0; i < LEN; i++) {
            if (hostA[i] == (hostB[i] + hostC[i]))
                continue;
            errors++;
        break;
        }
    }

    // Unmap the host buffers from device address space
    hipCheck(hipHostUnregister(hostC.get()));
    hipCheck(hipHostUnregister(hostB.get()));
    hipCheck(hipHostUnregister(hostA.get()));

    if (errors)
        std::cout << "FAILED" << std::endl;
    else
        std::cout << "PASSED" << std::endl;

    return errors;
}

int mainworker() {
    HipDevice hdevice;
    hdevice.showInfo(std::cout);

    hipFunction_t vaddfunction = hdevice.getFunction(FILENAME, KERNELNAME);
    hipFunction_t nopfunction = hdevice.getFunction(NOP_FILENAME, NOP_KERNELNAME);

    hipStream_t vaddstream;
    hipCheck(hipStreamCreateWithFlags(&vaddstream, hipStreamNonBlocking));

    hipStream_t nopstream;
    hipCheck(hipStreamCreateWithFlags(&nopstream, hipStreamNonBlocking));

    std::thread vaddthread = std::thread(mainworkerthread, vaddfunction, vaddstream, true);
    std::thread nopthread = std::thread(mainworkerthread, nopfunction, nopstream, false);

    vaddthread.join();
    nopthread.join();

//    mainworkerthread(vaddfunction, vaddstream, true);
//    mainworkerthread(nopfunction, nopstream, false);
    return 0;
}
}

int main()
{
    try {
        mainworker();

    } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}
