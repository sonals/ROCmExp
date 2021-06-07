/* SPDX-License-Identifier: Apache License 2.0 */
/* Copyright (c) 2021 Xilinx, Inc. All rights reserved */

#include <cstdio>
#include "hip/hip_runtime.h"

extern "C" __global__ void
vectoradd(float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ c)

{
    int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    a[i] = b[i] + c[i];
}
