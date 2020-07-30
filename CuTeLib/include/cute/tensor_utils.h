#pragma once

#include <cute/array.h>
#include <cute/tensor.h>
#include <numeric>
#include <random>

namespace cute
{


template <typename T, typename ShapesT>
[[nodiscard]] auto iota(ShapesT args, T initial = 0)
{
    auto result = Tensor<T, ShapesT::size(), Hardware::CPU>(std::move(args));
    std::iota(result.begin(), result.end(), initial);
    return result;
}

template <typename T, typename ShapesT>
[[nodiscard]] auto random(ShapesT args)
{
    auto result = Tensor<T, ShapesT::size(), Hardware::CPU>(std::move(args));
    std::generate(result.begin(), result.end(), std::rand);
    return result;
}

} // namespace cute