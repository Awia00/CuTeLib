#include <cute/array.h>
#include <cute/tensor_span.h>
#include <cute/unique_ptr.h>
#include <iostream>

namespace cute
{

void array_examples()
{
    // Example 01
    const auto i32_arr = Array<int32_t, 3>{ 0, 1, 2 };
    stream_array(std::cout << "i32_arr: ", i32_arr) << std::endl;
    stream_array(std::cout << "drop<1>: ", i32_arr.drop<1>()) << std::endl;
    stream_array(std::cout << "take<1>: ", i32_arr.take<1>()) << std::endl;
    stream_array(std::cout << "skip<1>: ", i32_arr.skip<1>()) << std::endl;
    std::cout << std::endl;

    // Example 02
    const auto i64_arr = Array<int64_t, 3>{ 1, 10, 100 };
    stream_array(std::cout << "i64_arr: ", i64_arr) << std::endl;
    std::cout << "product(): " << i64_arr.product() << std::endl;
    std::cout << "sum(): " << i64_arr.sum() << std::endl;
    std::cout << "inner_product(self): " << i64_arr.inner_product(i64_arr) << std::endl;
    std::cout << std::endl;
}

void unique_ptr_examples()
{
    // create a float cpu unique_ptr with 128 elements.
    // auto is std::unique_ptr<float[], HardwareDeleteFunctor<float, Hardware::CPU>>
    const auto uniq_cpu_ptr = cute::make_unique<float, Hardware::CPU>(128);

    // create a float cpu unique_ptr with 128 elements
    // auto is std::unique_ptr<float[], HardwareDeleteFunctor<float, Hardware::GPU>>
    const auto uniq_gpu_ptr = cute::make_unique<float, Hardware::GPU>(128);

    static_assert(!std::is_same_v<decltype(uniq_cpu_ptr), decltype(uniq_gpu_ptr)>,
                  "The two type are not the same Yay.");
    static_assert(std::is_same_v<decltype(uniq_cpu_ptr.get()), decltype(uniq_gpu_ptr.get())>,
                  ".get() is however the same type - so exercise regular care");

    // Notice that cute::make_unique returns a regular std::unique_ptr but with a custom deleter.
    //   Also notice that the template parameter T in unique_ptr is the array version T[].
    // You can also use the alias HardwareUniquePtr<T, Hardware>.

    // Both ofcourse gets deleted when running out of scope.
}

void tensor_span_examples()
{
    const auto cpu_data = cute::make_unique<float, Hardware::CPU>(128);
    auto vector_span = get_span_of_data(cpu_data, Array<int32_t, 1>{ 128 });
    vector_span.elem_ref(64) = 10;
    std::cout << "vector_span[0]:\t\t" << vector_span[0] << std::endl; // prints 0
    std::cout << "vector_span.elem(64):\t" << vector_span.elem(64) << std::endl; // prints 10

    auto matric_span = get_span_of_data(cpu_data, Array<int32_t, 2>{ 2, 64 });
    std::cout << "matric_span.elem(1,0):\t" << matric_span.elem(1, 0) << std::endl; // prints 10
    auto second_row_span = matric_span[1];
    std::cout << "matric_span[1].elem(0):\t" << second_row_span.elem(0) << std::endl; // prints 10

    // Notice that these can be used in kernels, as long as the Hardware template is GPU
}


} // namespace cute

int main()
{
    std::cout << "CuTeLib examples: " << std::endl;
    std::cout << std::endl;

    cute::array_examples();
    cute::unique_ptr_examples();
    cute::tensor_span_examples();
}