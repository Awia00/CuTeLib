#pragma once
#include <assert.h>
#include <cstdint>
#include <type_traits>


#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
#define CUTE_NVCC_DEVICE __device__
#define CUTE_NVCC_HOST __host__
#else
#define CUTE_NVCC_DEVICE
#define CUTE_NVCC_HOST
#endif

// combine host and dev
#define CUTE_DEV_HOST CUTE_NVCC_DEVICE CUTE_NVCC_HOST


#ifdef __CUDA_ARCH__
// I have not been successful in stopping the __device__ version being lazily compiled therefore we
// can only do an assert here. The compiler should however be able to remove it as it will always
// evaluate to true or false at compile time

//"Trying to access cpu elements on the gpu"
#define ENSURE_CORRECT_HARDWARE(MACRO_HARDWARE_VALUE) assert(MACRO_HARDWARE_VALUE == Hardware::GPU)
#else
// "Trying to access gpu elements on the cpu"
#define ENSURE_CORRECT_HARDWARE(MACRO_HARDWARE_VALUE) assert(MACRO_HARDWARE_VALUE == Hardware::CPU)
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

// Restrictiveness
#if defined(__CUDA_ARCH__) || defined(__CUDACC__) || defined(__GNUG__)
#define CUTE_RESTRICT __restrict__
#else
#define CUTE_RESTRICT __restrict
#endif


namespace cute
{

template <class... T>
constexpr bool always_false = false;

template <typename T, bool RestrictedV>
using restricted_ptr = typename std::conditional<RestrictedV, T * CUTE_RESTRICT, T*>::type;

}  // namespace cute
