# CuTeLib

## Purpose

The purpose of the **CU**DA **Te**mplate **Lib**rary is to provide constructs for better type safety and usability of CUDA code with no runtime overhead, both in kernels and in the calling host code.

## Examples

```c++
// Can be used in host or device function
#include <cute/array.h>

const auto i32_arr = cute::Array<int32_t, 3>{ 0, 1, 2 };
i32_arr.drop<1>()); //  Array<int32_t, 2>{ 0, 2 }
i32_arr.take<1>()); //  Array<int32_t, 1>{ 0 }
i32_arr.skip<1>()); //  Array<int32_t, 2>{ 1, 2 }

const auto i64_arr = cute::Array<int64_t, 3>{ 1, 10, 100 };
i64_arr.product();              // 1000
i64_arr.sum();                  // 111
i64_arr.inner_product(i64_arr); // 10101

```
