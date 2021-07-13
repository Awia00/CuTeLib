# CuTeLib

[![Build](https://github.com/anders-wind/CuTeLib/actions/workflows/ci.yml/badge.svg)](https://github.com/anders-wind/CuTeLib/actions/workflows/ci.yml)
[![CodeFactor](https://www.codefactor.io/repository/github/anders-wind/cutelib/badge)](https://www.codefactor.io/repository/github/anders-wind/cutelib)

## Purpose

The purpose of the **CU**DA **Te**mplate **Lib**rary is to provide constructs for better type safety and usability of CUDA code with no runtime overhead, both in kernels and in the calling host code.

```cpp
#include <cute/cute.h>

// The classic saxpy kernel, but using CuTeLib's Tensors
__global__ void saxpy(float a,
                      cute::TensorSpan<const float, 1, Hardware::GPU> x,
                      cute::TensorSpan<const float, 1, Hardware::GPU> y,
                      cute::TensorSpan<float, 1, Hardware::GPU> out)
{
    const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < x.shape<0>())
    {
        out[idx] = a * x[idx] + y[idx];
    }
}

void cutelib_intro()
{
    // generate a iota 1d tensor (0,1,2,...,31) and transfer to GPU
    auto x = cute::iota<float>(shape(32)).transfer<Hardware::GPU>();
    // generate a 1d tensor with random values and transfer to GPU
    auto y = cute::random<float>(shape(32)).transfer<Hardware::GPU>();
    // Allocate a 1d tensor 0 initialized directly on the GPU
    auto out = cute::Tensor<float, 1, Hardware::GPU>(cute::shape(32));

    // run kernel: notice implicit conversion for spans
    saxpy<<<1, 128>>>(0.5, x, y, out);

    // we can print and see the results
    std::cout << out.transfer<Hardware::CPU>() << std::endl;
}

```

For more have a look at the [Wiki](https://github.com/Awia00/CuTeLib/wiki)

## Install

Currently there are two ways to consume the library:

- Download the header files manually and add them to your build system
- Use [CPM.cmake](https://github.com/cpm-cmake/CPM.cmake) with `CPMAddPackage("gh:anders-wind/CuTeLib#3fd3d9d")`. Substitute `3fd3d9d` with your favorite hash or git tag (once added). At last add `cutelib` in your `target_link_library`

I wish to continue to support more dependency management frameworks, such as conan, in the future.

## Development

Build requirements (verified tested with):

- CUDA >= 11.3
- A C++17 or newer compatible compiler: MSVC >= 16.10 or GCC >= 9.3.0
- CMAKE >= 3.18
- PYTHON >= 3.7

You can build the library with the provided wrapper script `python build.py`.
