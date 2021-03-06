#pragma once

#include <limits>
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

static inline auto& default_random_engine_instance()
{
    static auto rand_gen = std::default_random_engine();
    return rand_gen;
}

/**
 * @brief Generates a tensor with random T values. Note its very simple and mostly meant for testing
 * and easy data generation. We use std::default_random_engine so exercise caution when using this
 *
 * @tparam T
 * @tparam ShapesT
 * @param args
 * @return auto
 */
template <typename T, typename ShapesT, typename RandomEngineT = std::default_random_engine>
[[nodiscard]] auto random(ShapesT args, RandomEngineT& rand_gen = default_random_engine_instance())
{
    auto result = Tensor<T, ShapesT::size(), Hardware::CPU>(std::move(args));
    if constexpr (std::is_floating_point_v<T>)
    {
        auto distri = std::uniform_real_distribution<T>();
        std::generate(result.begin(), result.end(), [&]() { return distri(rand_gen); });
    }
    else if constexpr (std::is_integral_v<T>)
    {
        auto distri = std::uniform_int_distribution<T>();
        std::generate(result.begin(), result.end(), [&]() { return distri(rand_gen); });
    }
    else
    {
        static_assert(always_false<T>, "cute::random does not support the given Type T for this function");
    }
    return result;
}

}  // namespace cute
