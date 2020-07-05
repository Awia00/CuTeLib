#pragma once
#include <assert.h>
#include <cstdint>


#if defined(__CUDA__ARCH__) || defined(__CUDACC__)
#define CUTE_NVCC_DEVICE __device__
#define CUTE_NVCC_HOST __host__
#else
#define CUTE_NVCC_DEVICE
#define CUTE_NVCC_HOST
#endif

#define CUTE_DEV_HOST CUTE_NVCC_DEVICE CUTE_NVCC_HOST


#ifdef __CUDA__ARCH__
#define ENSURE_CORRECT_HARDWARE(MACRO_HARDWARE_VALUE) \
    static_assert(MACRO_HARDWARE_VALUE == Hardware::GPU, "Trying to access cpu elements on the gpu")
#else
#define ENSURE_CORRECT_HARDWARE(MACRO_HARDWARE_VALUE) \
    static_assert(MACRO_HARDWARE_VALUE == Hardware::CPU, "Trying to access gpu elements on the cpu")
#endif


#ifdef __CUDACC__
#define ENSURE_CUDA_COMPILER_IF_GPU(MACRO_HARDWARE_VALUE) \
    do                                                    \
    {                                                     \
    } while (false)

#else
#define ENSURE_CUDA_COMPILER_IF_GPU(MACRO_HARDWARE_VALUE) \
    static_assert(MACRO_HARDWARE_VALUE == Hardware::CPU, "This function cannot be compiled without a CUDA compiler, make sure you template instantiate from a .cu file")
#endif
