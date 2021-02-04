
#include <cuda.h>
#include <cute/array.h>
#include <cute/tensor.h>
#include <cute/tensor_generators.h>
#include <cute/tensor_span.h>
#include <cute/unique_ptr.h>
#include <iostream>
#include <vector>

namespace cute
{


namespace /// example_id="cutelib_introduction"
{
#include <cuda.h>
#include <cute/array.h>
#include <cute/tensor.h>
#include <cute/tensor_generators.h>
#include <cute/tensor_span.h>
#include <cute/unique_ptr.h>

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

} // namespace

void array_examples()
{
    /// example_id="array_basics"
    {
        const auto i32_arr = cute::Array<int32_t, 3>{ 0, 1, 2 };

        // arrays can also be created with the variadic template function: array
        assert(i32_arr == cute::array(0, 1, 2));

        // array has a number of utility functions
        assert(i32_arr.take<1>() == cute::array(0));
        assert(i32_arr.skip<1>() == cute::array(1, 2));
        assert(i32_arr.drop<1>() == cute::array(0, 2));

        // You can print arrays using the stream_array function:
        std::cout << i32_arr << std::endl; // [0, 1, 2]
    }

    /// example_id="array_numerics"
    {
        const auto i64_arr = cute::Array<int64_t, 3>{ 1, 10, 100 };

        assert(i64_arr.mul() == 1 * 10 * 100);
        assert(i64_arr.sum() == 1 + 10 + 100);
        assert(i64_arr.dot_product(i64_arr) == 1 + 100 + 10000);
    }
}

void unique_ptr_examples()
{
    // create a float cpu unique_ptr with 128 elements.
    // auto is std::unique_ptr<float[], HardwareDeleteFunctor<float, Hardware::CPU>>
    const auto uniq_cpu_ptr = cute::make_unique<float[], Hardware::CPU>(128);

    // create a float cpu unique_ptr with 128 elements
    // auto is std::unique_ptr<float[], HardwareDeleteFunctor<float, Hardware::GPU>>
    const auto uniq_gpu_ptr = cute::make_unique<float[], Hardware::GPU>(128);

    static_assert(!std::is_same_v<decltype(uniq_cpu_ptr), decltype(uniq_gpu_ptr)>,
                  "The two type are not the same Yay.");
    static_assert(std::is_same_v<decltype(uniq_cpu_ptr.get()), decltype(uniq_gpu_ptr.get())>,
                  ".get() is however the same type - so exercise regular care");

    // Notice that cute::make_unique returns a regular std::unique_ptr but with a custom deleter.
    //   Also notice that the template parameter T in unique_ptr is the array version T[].
    // You can also use the alias HardwareUniquePtr<T, Hardware>.

    // Both of course gets deleted when running out of scope.
}

void tensor_span_examples()
{
    // cute::make_unique always wants array types for now.
    auto cpu_data = cute::make_unique<float[], Hardware::CPU>(128);
    auto vector_span = get_span_of(cpu_data, shape(128));
    vector_span.elem_ref(0) = 0;
    vector_span.elem_ref(64) = 10;
    std::cout << "vector_span[0]:\t\t" << vector_span[0] << std::endl; // prints 0
    std::cout << "vector_span.elem(64):\t" << vector_span.elem(64) << std::endl; // prints 10

    auto matrix_span = get_span_of(cpu_data, shape(2, 64));
    std::cout << "matrix_span.elem(1, 0):\t" << matrix_span.elem(1, 0) << std::endl; // prints 10
    auto second_row_span = matrix_span[1];
    std::cout << "matrix_span[1].elem(0):\t" << second_row_span.elem(0) << std::endl; // prints 10

    // You can use spans over raw data, or you can use the Tensor class which owns data and allows you to easily get a span.
    auto tensor = Tensor<int8_t, 3, Hardware::CPU>({ 1, 2, 3 });
    auto tensor_span = tensor.get_span();
    auto& elem = tensor_span.elem_ref(0, 1, 2);
    elem = 1;
    std::cout << "span.elem_ref(0, 1, 2):\t" << static_cast<bool>(tensor_span.elem_ref(0, 1, 2))
              << std::endl; // prints 1

    // or const. Note that spans over constant data does not have to be constant themselves.
    const auto tensor_const = Vector<double, Hardware::CPU>(std::vector<double>{ 5, 4, 3, 2, 1 }, { 5 }); // Vector is just a 1d Tensor
    auto tensor_const_span = tensor_const.get_span();
    const auto& elem_const = tensor_const_span.elem_ref(1);
    std::cout << "const_span:\t" << elem_const << std::endl; // prints 4.0


    // We can use the copy constructors or use the transfer methods for transfering data to and from different hardware.
    auto cpu_tensor = Tensor<int32_t, 2, Hardware::CPU>({ 2, 2 });
    auto cpu_transfered = cpu_tensor.transfer<Hardware::CPU>(); // just a regular copy
    cpu_transfered.get_span().elem_ref(0, 0) = 123;

    // there...
    auto gpu_transfered = cpu_transfered.transfer<Hardware::GPU>();
    cpu_transfered.get_span().elem_ref(0, 0) = 1; // zero out to show it is a copy

    // ... and back again
    auto moved_back = gpu_transfered.transfer<Hardware::CPU>();
    std::cout << "CPU->GPU->CPU:\t" << moved_back.get_span().elem(0, 0) << std::endl; // prints 123
    std::cout << std::endl;
}


__global__ void my_kernel(TensorSpan<const float, 2, Hardware::GPU> x,
                          TensorSpan<const float, 1, Hardware::GPU> y,
                          TensorSpan<float, 1, Hardware::GPU> out)
{
    auto idx = threadIdx.x + blockDim.x * blockIdx.x;
    auto res = 0.0f;

    if (idx < x.shape(1) && idx < y.shape(0))
    {
        auto multiplier = y.elem(idx);
        for (auto i = 0; i < x.shape(0); i++)
        {
            res += x.elem(i, idx) * multiplier;
        }
    }

    if (idx < out.shape(0))
    {
        out.elem_ref(idx) = res;
    }
}

void kernel_use_example()
{
    const auto x = iota<float>(shape(18, 32)).transfer<Hardware::GPU>();
    const auto y = iota<float>(shape(32)).transfer<Hardware::GPU>();
    auto out = Tensor<float, 1, Hardware::GPU>(shape(32));
    my_kernel<<<1, 128>>>(x.get_span(), y.get_span(), out.get_span());

    std::cout << out.transfer<Hardware::CPU>() << std::endl;
}

} // namespace cute

int main()
{
    std::cout << "CuTeLib examples: " << std::endl;
    std::cout << std::endl;

    cute::cutelib_intro();
    cute::array_examples();
    cute::unique_ptr_examples();
    cute::tensor_span_examples();
    cute::kernel_use_example();

    std::cout << "Done" << std::endl;
    return 0;
}