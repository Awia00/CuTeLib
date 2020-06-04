#pragma once
#include <assert.h>
#include <cstdint>


#define CUTE_NVCC_DEVICE __device__
#define CUTE_NVCC_HOST __host__

#define CUTE_DEV_HOST CUTE_NVCC_DEVICE CUTE_NVCC_HOST