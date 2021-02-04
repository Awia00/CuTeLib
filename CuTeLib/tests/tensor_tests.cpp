#include "test_utils.h"
#include <cute/stream.h>
#include <cute/tensor.h>
#include <doctest/doctest.h>

namespace cute
{

TEST_SUITE("Tensor")
{
    TEST_CASE("vector_constructor")
    {
        // or const. Note that spans over constant data does not have to be constant themselves.
        auto tensor_const = Vector<double, Hardware::CPU>(std::vector<double>{ 5, 4, 3, 2, 1 }, shape(5)); // Vector is just an alias for 1d Tensor
        auto expect_tensor = Tensor<double, 1, Hardware::CPU>(shape(5));
        expect_tensor.get_span().elem_ref(0) = 5;
        expect_tensor.get_span().elem_ref(1) = 4;
        expect_tensor.get_span().elem_ref(2) = 3;
        expect_tensor.get_span().elem_ref(3) = 2;
        expect_tensor.get_span().elem_ref(4) = 1;
        check_tensors(tensor_const, expect_tensor);
    }
}

} // namespace cute
