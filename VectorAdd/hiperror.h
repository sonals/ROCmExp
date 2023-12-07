/* SPDX-License-Identifier: Apache License 2.0 */
/* Copyright (C) 2023 Advanced Micro Devices, Inc. */

#include <stdexcept>
#include <string>
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

inline void errorCheck(hipError_t status, const char *note = "") {
    if (status != hipSuccess) {       \
        throw HIPError(status, note);   \
    }
}
