#pragma once

#include <numeric>
#include <random>
#include <utility>
#include <cute/array.h>
#include <cute/tensor.h>

namespace cute
{


template <typename T, typename ShapesT>
[[nodiscard]] auto iota(ShapesT args, T initial = 0)
{
    auto result = Tensor<T, ShapesT::size(), Hardware::CPU>(std::forward<ShapesT>(args));
    std::iota(result.begin(), result.end(), initial);
    return result;
}

/**
 * @brief Generates a tensor with random T values. Note we only use std::rand and therefore this
 * should not be used when truly random distribution is required.
 *
 * @tparam T
 * @tparam ShapesT
 * @param args
 * @return auto
 */
template <typename T, typename ShapesT>
[[nodiscard]] auto random(ShapesT args)
{
    auto result = Tensor<T, ShapesT::size(), Hardware::CPU>(std::move(args));
    std::generate(result.begin(),
                  result.end(),
                  []()
                  {
                      if constexpr (std::is_floating_point_v<T>)
                      {
                          return std::rand() / T(RAND_MAX);
                      }
                      else
                      {
                          return T(std::rand());
                      }
                  });
    return result;
}

}  // namespace cute
