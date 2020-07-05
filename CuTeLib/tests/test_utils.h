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
    CHECK(tensor.size() == expected_tensor.size());
    res &= tensor.size() == expected_tensor.size();
    for (auto i = 0; i < tensor.size(); i++)
    {
        INFO(i);

        auto value = tensor.data()[i];
        auto expected_value = expected_tensor.data()[i];
        if (std::is_floating_point_v<T>)
        {
            // https://stackoverflow.com/a/21603815
            constexpr auto ulp = 7;
            auto are_equal = std::abs(value - expected_value) <=
                             std::numeric_limits<T>::epsilon() *
                                 std::max(std::abs(value), std::abs(expected_value)) * ulp;
            CHECK(are_equal);
            res &= are_equal;
        }
        else
        {
            CHECK(value == expected_value);
            res &= value == expected_value;
        }
    }
    return res;
}

} // namespace cute