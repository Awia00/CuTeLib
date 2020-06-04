#include <cute/array.h>
#include <iostream>

namespace cute
{

/// Namespaced run
void run()
{
    std::cout << "CuTeLib examples: " << std::endl;
    std::cout << std::endl;

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


} // namespace cute

int main()
{
    cute::run();
}