#pragma once
#include <cuda.h>
#include <cute/hardware.h>
#include <memory>
#include <vector>

namespace cute
{


template <typename T>
struct DeleteFunctorGPU
{
    constexpr DeleteFunctorGPU() noexcept = default;
    using TBase = typename std::remove_all_extents_t<T>;

    void operator()(TBase* p) const noexcept
    {
        static_assert(0 < sizeof(TBase), "can't delete an incomplete type");
        cudaFree(p);
    }
};

template <typename T>
struct NewFunctorGPU
{
    constexpr NewFunctorGPU() noexcept = default;

    [[nodiscard]] auto operator()(size_t num_elements) const noexcept
    {
        using TBase = typename std::remove_all_extents_t<T>;
        TBase* ptr;
        cudaMalloc(&ptr, num_elements * sizeof(float));
        return ptr;
    }
};

template <typename T>
struct NewFunctorCPU
{
    constexpr NewFunctorCPU() noexcept = default;
    using TBase = typename std::remove_all_extents_t<T>;

    [[nodiscard]] auto operator()(size_t num_elements) noexcept
    {
        return new TBase[num_elements];
    }
};

template <typename T, Hardware HardwareV>
using HardwareDeleteFunctor =
    typename std::conditional_t<HardwareV == Hardware::GPU, DeleteFunctorGPU<T>, std::default_delete<T>>;

template <typename T, Hardware HardwareV>
using HardwareNewFunctor =
    typename std::conditional_t<HardwareV == Hardware::GPU, NewFunctorGPU<T>, NewFunctorCPU<T>>;


template <typename T, Hardware HardwareV>
using HardwareUniquePtr = std::unique_ptr<T, HardwareDeleteFunctor<T, HardwareV>>;

template <typename T, Hardware HardwareV>
HardwareUniquePtr<T, HardwareV> make_unique(size_t num_elements)
{
    static_assert(std::is_array_v<T>, "Must be array type");
    return HardwareUniquePtr<T, HardwareV>(HardwareNewFunctor<T, HardwareV>()(num_elements));
}

template <typename HardwareUniquePtrT>
constexpr inline Hardware what_hardware() noexcept
{
    using RemRef = typename std::remove_reference_t<HardwareUniquePtrT>;
    using T = typename std::remove_const_t<typename RemRef::element_type>;
    using DT = typename RemRef::deleter_type;
    return std::is_same_v<DT, DeleteFunctorGPU<T[]>> ? Hardware::GPU : Hardware::CPU;
}


enum struct MemcpyType
{
    HostToHost,
    HostToDevice,
    DeviceToHost,
    DeviceToDevice
};

template <Hardware HardwareFromV, Hardware HardwareToV>
constexpr static MemcpyType get_memcpy_type()
{
    if (HardwareFromV == HardwareToV)
    {
        if (HardwareFromV == Hardware::CPU)
        {
            return MemcpyType::HostToHost;
        }
        return MemcpyType::DeviceToDevice;
    }
    if (HardwareFromV == Hardware::CPU)
    {
        return MemcpyType::HostToDevice;
    }
    return MemcpyType::DeviceToHost;
}

template <typename HardwareUniquePtrFromT, typename HardwareUniquePtrToT>
void memcpy(const HardwareUniquePtrFromT& from_ptr, HardwareUniquePtrToT& to_ptr, size_t elements)
{
    using FromRemRef = typename std::remove_reference_t<HardwareUniquePtrFromT>;
    using ToRemRef = typename std::remove_reference_t<HardwareUniquePtrToT>;
    using FromT = typename std::remove_const_t<typename FromRemRef::element_type>;
    using T = typename std::remove_const_t<typename ToRemRef::element_type>;
    static_assert(std::is_same_v<FromT, T>, "From and to were not of same type");

    constexpr auto memcpy_type = get_memcpy_type<what_hardware<FromRemRef>(), what_hardware<ToRemRef>()>();

    // Since memcpy_type is constexpr, these if statements are optimized away - but I will not add if-constexpr for now since it is only supported in CUDA 11.
    if (memcpy_type == MemcpyType::HostToHost)
    {
        std::copy(from_ptr.get(), from_ptr.get() + elements, to_ptr.get());
    }
    else if (memcpy_type == MemcpyType::HostToDevice)
    {
        cudaMemcpy(to_ptr.get(), from_ptr.get(), elements * sizeof(T), cudaMemcpyHostToDevice);
    }
    else if (memcpy_type == MemcpyType::DeviceToHost)
    {
        cudaMemcpy(to_ptr.get(), from_ptr.get(), elements * sizeof(T), cudaMemcpyDeviceToHost);
    }
    else // if (memcpy_type == MemcpyType::DeviceToDevice)
    {
        cudaMemcpy(to_ptr.get(), from_ptr.get(), elements * sizeof(T), cudaMemcpyDeviceToDevice);
    }
}

template <typename FromT, typename HardwareUniquePtrToT>
void memcpy(const std::vector<FromT>& from_ptr, HardwareUniquePtrToT& to_ptr, size_t elements)
{
    using ToRemRef = typename std::remove_reference_t<HardwareUniquePtrToT>;
    using T = typename std::remove_const_t<typename ToRemRef::element_type>;
    static_assert(std::is_same_v<FromT, T>, "From and to were not of same type");

    const auto from_data = from_ptr.data();

    constexpr auto memcpy_type = get_memcpy_type<Hardware::CPU, what_hardware<HardwareUniquePtrToT>()>();

    // Since memcpy_type is constexpr, these if statements are optimized away - but I will not add if-constexpr for now since it is only supported in CUDA 11.
    if (memcpy_type == MemcpyType::HostToHost)
    {
        std::copy(from_data, from_data + elements, to_ptr.get());
    }
    else if (memcpy_type == MemcpyType::HostToDevice)
    {
        cudaMemcpy(to_ptr.get(), from_data, elements * sizeof(T), cudaMemcpyHostToDevice);
    }
    else if (memcpy_type == MemcpyType::DeviceToHost)
    {
        cudaMemcpy(to_ptr.get(), from_data, elements * sizeof(T), cudaMemcpyDeviceToHost);
    }
    else // if (memcpy_type == MemcpyType::DeviceToDevice)
    {
        cudaMemcpy(to_ptr.get(), from_data, elements * sizeof(T), cudaMemcpyDeviceToDevice);
    }
}

template <typename HardwareUniquePtrT, typename T>
void memset(HardwareUniquePtrT& ptr, T val, size_t num_bytes)
{
    using RemRef = typename std::remove_reference_t<HardwareUniquePtrT>;
    constexpr auto hardware = what_hardware<RemRef>();

    if (hardware == Hardware::CPU)
    {
        std::memset(ptr.get(), val, num_bytes);
    }
    else // if (hardware == MemcpyType::HostToDevice)
    {
        cudaMemset(ptr.get(), val, num_bytes);
    }
}

} // namespace cute