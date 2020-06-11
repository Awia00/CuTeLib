#pragma once
#include <cuda.h>
#include <cute/hardware.h>
#include <memory>

namespace cute
{

template <typename T>
struct DeleteFunctorGPU
{
    constexpr DeleteFunctorGPU() noexcept = default;

    void operator()(T* p) const noexcept
    {
        cudaFree(p);
    }
};

template <typename T>
struct NewFunctorGPU
{
    constexpr NewFunctorGPU() noexcept = default;

    [[nodiscard]] T* operator()(size_t num_elements) const noexcept
    {
        T* ptr;
        cudaMalloc(&ptr, num_elements * sizeof(float));
        cudaMemset(&ptr, T(), num_elements * sizeof(float));
        return ptr;
    }
};

template <typename T>
struct NewFunctorCPU
{
    constexpr NewFunctorCPU() noexcept = default;

    [[nodiscard]] T* operator()(size_t num_elements) noexcept
    {
        return new T[num_elements]();
    }
};

template <typename T, Hardware HardwareV>
using HardwareDeleteFunctor =
    typename std::conditional_t<HardwareV == Hardware::GPU, DeleteFunctorGPU<T>, std::default_delete<T[]>>;

template <typename T, Hardware HardwareV>
using HardwareNewFunctor =
    typename std::conditional_t<HardwareV == Hardware::GPU, NewFunctorGPU<T>, NewFunctorCPU<T>>;


template <typename T, Hardware HardwareV>
using HardwareUniquePtr = std::unique_ptr<T[], HardwareDeleteFunctor<T, HardwareV>>;

template <typename T, Hardware HardwareV>
HardwareUniquePtr<T, HardwareV> make_unique(size_t num_elements)
{
    static_assert(!std::is_array_v<T>, "Does not support array array types");
    return HardwareUniquePtr<T, HardwareV>(HardwareNewFunctor<T, HardwareV>()(num_elements));
}


template <typename HardwareUniquePtrT>
constexpr inline Hardware what_hardware() noexcept
{
    using RemRef = typename std::remove_reference_t<HardwareUniquePtrT>;
    using T = typename std::remove_const_t<typename RemRef::element_type>;
    using DT = typename RemRef::deleter_type;
    return std::is_same_v<DT, DeleteFunctorGPU<T>> ? Hardware::GPU : Hardware::CPU;
}

} // namespace cute