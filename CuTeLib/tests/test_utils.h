#include <algorithm>
#include <cute/tensor.h>
#include <doctest/doctest.h>

namespace cute
{


template <typename TensorT, typename OtherTensorT>
bool check_tensors(const TensorT& tensor, const OtherTensorT& expected_tensor)
{
    static_assert(TensorT::hardware() == Hardware::CPU, "Can only test CPU tensors");
    static_assert(OtherTensorT::hardware() == Hardware::CPU, "Can only test CPU tensors");
    static_assert(std::is_same_v<typename TensorT::value_type, typename OtherTensorT::value_type>,
                  "value_types must be the same");
    using T = typename TensorT::value_type;

    auto res = true;
    CHECK_EQ(tensor.size(), expected_tensor.size());
    res &= tensor.size() == expected_tensor.size();
    for (auto i = 0; i < tensor.size(); i++)
    {
        INFO(i);
        auto value = tensor.data()[i];
        auto expected_value = expected_tensor.data()[i];
        if constexpr (std::is_floating_point_v<T>)
        {
            CHECK_EQ(value, doctest::Approx(expected_value));
        }
        else
        {
            CHECK_EQ(value, expected_value);
            res &= value == expected_value;
        }
    }
    return res;
}

}  // namespace cute
