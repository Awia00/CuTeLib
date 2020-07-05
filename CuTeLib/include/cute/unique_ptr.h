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

    void operator()(TBase* p) const noexcept;
};

template <typename T>
struct NewFunctorGPU
{
    constexpr NewFunctorGPU() noexcept = default;

    [[nodiscard]] auto operator()(size_t num_elements) const noexcept;
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

#ifdef __CUDACC__
template <typename T>
void DeleteFunctorGPU<T>::operator()(TBase* p) const noexcept
{
    static_assert(0 < sizeof(TBase), "can't delete an incomplete type");
    cudaFree(p);
}


template <typename T>
[[nodiscard]] auto NewFunctorGPU<T>::operator()(size_t num_elements) const noexcept
{
    using TBase = typename std::remove_all_extents_t<T>;
    TBase* ptr;
    cudaMalloc(&ptr, num_elements * sizeof(float));
    return ptr;
}
#endif

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
    ENSURE_CUDA_COMPILER_IF_GPU(HardwareV);
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


template <MemcpyType MemcpyTypeT>
struct MemCpyPartialTemplateSpecializer
{
    template <typename T>
    static void memcpy_data(const T* from_ptr, T* to_ptr, size_t elements);
};

#ifdef __CUDACC__
template <MemcpyType MemcpyTypeT>
template <typename T>
void MemCpyPartialTemplateSpecializer<MemcpyTypeT>::memcpy_data<T>(const T* from_ptr, T* to_ptr, size_t elements)
{
    if (MemcpyTypeT == MemcpyType::HostToHost)
    {
        std::copy(from_ptr, from_ptr + elements, to_ptr);
    }
    else if (MemcpyTypeT == MemcpyType::HostToDevice)
    {
        cudaMemcpy(to_ptr, from_ptr, elements * sizeof(T), cudaMemcpyHostToDevice);
    }
    else if (MemcpyTypeT == MemcpyType::DeviceToHost)
    {
        cudaMemcpy(to_ptr, from_ptr, elements * sizeof(T), cudaMemcpyDeviceToHost);
    }
    else // if (MemcpyTypeT == MemcpyType::DeviceToDevice)
    {
        cudaMemcpy(to_ptr, from_ptr, elements * sizeof(T), cudaMemcpyDeviceToDevice);
    }
}

#else
template <>
struct MemCpyPartialTemplateSpecializer<MemcpyType::HostToHost>
{
    template <typename T>
    static void memcpy_data(const T* from_ptr, T* to_ptr, size_t elements)
    {
        std::copy(from_ptr, from_ptr + elements, to_ptr);
    }
};

#endif


template <typename HardwareUniquePtrFromT, typename HardwareUniquePtrToT>
void memcpy(const HardwareUniquePtrFromT& from_ptr, HardwareUniquePtrToT& to_ptr, size_t elements)
{
    using FromRemRef = typename std::remove_reference_t<HardwareUniquePtrFromT>;
    using ToRemRef = typename std::remove_reference_t<HardwareUniquePtrToT>;
    using FromT = typename std::remove_const_t<typename FromRemRef::element_type>;
    using T = typename std::remove_const_t<typename ToRemRef::element_type>;
    static_assert(std::is_same_v<FromT, T>, "From and to were not of same type");

    constexpr auto memcpy_type = get_memcpy_type<what_hardware<FromRemRef>(), what_hardware<ToRemRef>()>();
    MemCpyPartialTemplateSpecializer<memcpy_type>::memcpy_data(from_ptr.get(), to_ptr.get(), elements);
}

template <typename FromT, typename HardwareUniquePtrToT>
void memcpy(const std::vector<FromT>& from_ptr, HardwareUniquePtrToT& to_ptr, size_t elements)
{
    using ToRemRef = typename std::remove_reference_t<HardwareUniquePtrToT>;
    using T = typename std::remove_const_t<typename ToRemRef::element_type>;
    static_assert(std::is_same_v<FromT, T>, "From and to were not of same type");

    constexpr auto memcpy_type = get_memcpy_type<Hardware::CPU, what_hardware<HardwareUniquePtrToT>()>();
    MemCpyPartialTemplateSpecializer<memcpy_type>::memcpy_data(from_ptr.data(), to_ptr.get(), elements);
}


template <Hardware HardwareV>
struct MemsetPartialTemplateSpecializer
{
    template <typename T>
    static void memset_data(T* ptr, T val, size_t num_bytes);
};

#ifdef __CUDACC__
template <Hardware HardwareV>
template <typename T>
void MemsetPartialTemplateSpecializer<HardwareV>::memset_data<T>(T* ptr, T val, size_t num_bytes)
{
    if (HardwareV == Hardware::CPU)
    {
        std::memset(ptr, val, num_bytes);
    }
    else // if (hardware == MemcpyType::HostToDevice)
    {
        cudaMemset(ptr, val, num_bytes);
    }
}

#else
template <>
struct MemsetPartialTemplateSpecializer<Hardware::CPU>
{
    template <typename T>
    static void memset_data(T* ptr, T val, size_t num_bytes)
    {
        std::memset(ptr, val, num_bytes);
    }
};
#endif


template <typename HardwareUniquePtrT, typename T>
void memset(HardwareUniquePtrT& ptr, T val, size_t num_bytes)
{
    using RemRef = typename std::remove_reference_t<HardwareUniquePtrT>;
    constexpr auto hardware = what_hardware<RemRef>();

    MemsetPartialTemplateSpecializer<hardware>::memset_data(ptr.get(), val, num_bytes);
}


} // namespace cute