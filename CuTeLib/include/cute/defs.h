#pragma once
#include <assert.h>
#include <cstdint>


#define CUTE_NVCC_DEVICE __device__
#define CUTE_NVCC_HOST __host__

#define CUTE_DEV_HOST CUTE_NVCC_DEVICE CUTE_NVCC_HOST


#ifdef __CUDA__ARCH__
#define ENSURE_CORRECT_HARDWARE(MACRO_HARDWARE_VALUE) \
    static_assert(MACRO_HARDWARE_VALUE == Hardware::GPU, "Trying to access cpu elements on the gpu")
#else
#define ENSURE_CORRECT_HARDWARE(MACRO_HARDWARE_VALUE) \
    static_assert(MACRO_HARDWARE_VALUE == Hardware::CPU, "Trying to access gpu elements on the cpu")
#endif
