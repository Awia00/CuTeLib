# CuTeLib

[![Build](https://github.com/anders-wind/CuTeLib/actions/workflows/ci.yml/badge.svg)](https://github.com/anders-wind/CuTeLib/actions/workflows/ci.yml)

## Purpose

The purpose of the **CU**DA **Te**mplate **Lib**rary is to provide constructs for better type safety and usability of CUDA code with no runtime overhead, both in kernels and in the calling host code.

```cpp
#include <cuda.h>
#include <cute/array.h>
#include <cute/tensor.h>
#include <cute/tensor_generators.h>
#include <cute/tensor_span.h>

__global__ void saxpy(float a,
                      cute::TensorSpan<const float, 1, Hardware::GPU> x,
                      cute::TensorSpan<const float, 1, Hardware::GPU> y,
                      cute::TensorSpan<float, 1, Hardware::GPU> out)
{
    const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < x.shape<0>()) // idx is valid given x's size on the first dimension
    {
        out[idx] = a * x[idx] + y[idx];
    }
}

void cutelib_intro()
{
    // generate a iota 1d tensor (0,1,2,...,32) and transfer to GPU
    const auto x = cute::iota<float>(shape(32)).transfer<Hardware::GPU>();
    // generate a 1d tensor with random values and transfer to GPU
    const auto y = cute::random<float>(shape(32)).transfer<Hardware::GPU>();
    // Allocate a 1d tensor 0 initialized directly on the GPU
    auto out = cute::Tensor<float, 1, Hardware::GPU>(cute::shape(32));

    // run kernel: notice implicit conversion for spans (get_span if you want to be explicit)
    saxpy<<<1, 128>>>(0.5, x, y, out);

    // we can print and see the results
    std::cout << out.transfer<Hardware::CPU>() << std::endl;
}

```

For more have a look at the [Wiki](https://github.com/Awia00/CuTeLib/wiki)


## Development

Build requirements:

- CUDA >= 11.3
- MSVC >= 16.10
- CMAKE >= 3.18
