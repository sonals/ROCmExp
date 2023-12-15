/* SPDX-License-Identifier: Apache License 2.0 */
/* Copyright (C) 2023 Advanced Micro Devices, Inc. */

#include <stdexcept>
#include <string>
#include <chrono>
#include <system_error>

#include "hip/hip_runtime_api.h"

class HIPError : public std::system_error
{
private:
    static std::string message(hipError_t ec, const std::string& what) {
        std::string str = what;
        str += ": ";
        str += hipGetErrorString(ec);
        str += " (";
        str += hipGetErrorName(ec);
        str += ")";
        return str;
//        hipDrvGetErrorString(result, str);
    }

public:
  explicit
  HIPError(hipError_t ec, const std::string& what = "")
      : system_error(ec, std::system_category(), message(ec, what))
  {}
};

inline void hipCheck(hipError_t status, const char *note = "") {
    if (status != hipSuccess) {       \
        throw HIPError(status, note);   \
    }
}


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
template<typename T> class DeviceBO {
    T *_buffer;
public:
    DeviceBO(size_t size) : _buffer(nullptr) {
        hipCheck(hipMalloc((void**)&_buffer, size * sizeof(T)));
    }
    ~DeviceBO() noexcept {
        hipCheck(hipFree(_buffer));
    }
    T *get() const {
        return _buffer;
    }

    T *&get() {
        return _buffer;
    }

};
