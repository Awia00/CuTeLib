#pragma once
#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>
#include <cute/hardware.h>
#include <cute/stream.h>
#ifdef __CUDACC__
#include <cuda.h>
#endif

namespace cute
{


template <typename T>
struct DeleteFunctorGPU
{
    constexpr DeleteFunctorGPU() noexcept = default;
    using TBase = typename std::remove_all_extents_t<T>;

    constexpr void operator()(TBase* p) const noexcept;
};

template <typename T>
struct NewFunctorGPU
{
    constexpr NewFunctorGPU() noexcept = default;

    [[nodiscard]] constexpr auto operator()(size_t num_elements) const noexcept;
    [[nodiscard]] constexpr auto operator()(size_t num_elements, StreamView& stream) const noexcept;
};

template <typename T>
struct NewFunctorCPU
{
    constexpr NewFunctorCPU() noexcept = default;
    using TBase = typename std::remove_all_extents_t<T>;

    [[nodiscard]] constexpr auto operator()(size_t num_elements) noexcept
    {
        return new TBase[num_elements];
    }

    [[nodiscard]] constexpr auto operator()(size_t num_elements, StreamView& stream) noexcept
    {
        return new TBase[num_elements];
    }
};

#ifdef __CUDACC__
template <typename T>
constexpr void DeleteFunctorGPU<T>::operator()(TBase* p) const noexcept
{
    static_assert(0 < sizeof(TBase), "can't delete an incomplete type");
    cudaFree(p);
}

template <typename T>
[[nodiscard]] constexpr auto NewFunctorGPU<T>::operator()(size_t num_elements) const noexcept
{
    using TBase = typename std::remove_all_extents_t<T>;
    TBase* ptr;
    cudaMalloc(&ptr, num_elements * sizeof(float));
    return ptr;
}

template <typename T>
[[nodiscard]] constexpr auto NewFunctorGPU<T>::operator()(size_t num_elements, StreamView& stream) const noexcept
{
    using TBase = typename std::remove_all_extents_t<T>;
    TBase* ptr;
    cudaMallocAsync(&ptr, num_elements * sizeof(float), stream);
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
[[nodiscard]] constexpr HardwareUniquePtr<T, HardwareV> make_unique(size_t num_elements)
{
    static_assert(std::is_array_v<T>, "Must be array type");  // if not make_unique should receive arguments to construct T
    return HardwareUniquePtr<T, HardwareV>(HardwareNewFunctor<T, HardwareV>()(num_elements));
}

template <typename T, Hardware HardwareV>
[[nodiscard]] constexpr HardwareUniquePtr<T, HardwareV> make_unique_async(size_t num_elements, StreamView& stream)
{
    static_assert(std::is_array_v<T>, "Must be array type");  // if not make_unique should receive arguments to construct T
    return HardwareUniquePtr<T, HardwareV>(HardwareNewFunctor<T, HardwareV>()(num_elements, stream));
}

template <typename HardwareUniquePtrT>
[[nodiscard]] constexpr Hardware what_hardware() noexcept
{
    using RemRef = typename std::remove_reference_t<HardwareUniquePtrT>;
    using T = typename std::remove_const_t<typename RemRef::element_type>;
    using DT = typename RemRef::deleter_type;
    return std::is_same_v<DT, DeleteFunctorGPU<T[]>> ? Hardware::GPU : Hardware::CPU;
}


// =============== MEMCPY ===============

enum struct MemcpyType
{
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3
};

template <Hardware HardwareFromV, Hardware HardwareToV>
[[nodiscard]] constexpr static MemcpyType get_memcpy_type()
{
    if constexpr (HardwareFromV == HardwareToV)
    {
        if constexpr (HardwareFromV == Hardware::CPU)
        {
            return MemcpyType::HostToHost;
        }
        return MemcpyType::DeviceToDevice;
    }
    if constexpr (HardwareFromV == Hardware::CPU)
    {
        return MemcpyType::HostToDevice;
    }
    return MemcpyType::DeviceToHost;
}

template <MemcpyType MemcpyTypeT>
struct MemCpyPartialTemplateSpecializer
{
    template <typename T>
    constexpr static void memcpy_data(const T* from_ptr, T* to_ptr, size_t elements);
    template <typename T>
    constexpr static void memcpy_data_async(const T* from_ptr, T* to_ptr, size_t elements, StreamView& stream);
};

#ifdef __CUDACC__
template <MemcpyType MemcpyTypeT>
template <typename T>
constexpr void MemCpyPartialTemplateSpecializer<MemcpyTypeT>::memcpy_data<T>(const T* from_ptr, T* to_ptr, size_t elements)
{
    if constexpr (MemcpyTypeT == MemcpyType::HostToHost)
    {
        std::copy(from_ptr, from_ptr + elements, to_ptr);
    }
    else if constexpr (MemcpyTypeT == MemcpyType::HostToDevice)
    {
        cudaMemcpy(to_ptr, from_ptr, elements * sizeof(T), cudaMemcpyHostToDevice);
    }
    else if constexpr (MemcpyTypeT == MemcpyType::DeviceToHost)
    {
        cudaMemcpy(to_ptr, from_ptr, elements * sizeof(T), cudaMemcpyDeviceToHost);
    }
    else  // if (MemcpyTypeT == MemcpyType::DeviceToDevice)
    {
        cudaMemcpy(to_ptr, from_ptr, elements * sizeof(T), cudaMemcpyDeviceToDevice);
    }
}

template <MemcpyType MemcpyTypeT>
template <typename T>
constexpr void MemCpyPartialTemplateSpecializer<MemcpyTypeT>::memcpy_data_async<T>(const T* from_ptr,
                                                                                   T* to_ptr,
                                                                                   size_t elements,
                                                                                   StreamView& stream)
{
    if constexpr (MemcpyTypeT == MemcpyType::HostToHost)
    {
        std::copy(from_ptr, from_ptr + elements, to_ptr);
    }
    else if constexpr (MemcpyTypeT == MemcpyType::HostToDevice)
    {
        cudaMemcpyAsync(to_ptr, from_ptr, elements * sizeof(T), cudaMemcpyHostToDevice, stream);
    }
    else if constexpr (MemcpyTypeT == MemcpyType::DeviceToHost)
    {
        cudaMemcpyAsync(to_ptr, from_ptr, elements * sizeof(T), cudaMemcpyDeviceToHost, stream);
    }
    else  // if (MemcpyTypeT == MemcpyType::DeviceToDevice)
    {
        cudaMemcpyAsync(to_ptr, from_ptr, elements * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    }
}

#else
template <>
struct MemCpyPartialTemplateSpecializer<MemcpyType::HostToHost>
{
    template <typename T>
    constexpr static void memcpy_data(const T* from_ptr, T* to_ptr, size_t elements)
    {
        std::copy(from_ptr, from_ptr + elements, to_ptr);
    }

    template <typename T>
    constexpr static void memcpy_data_async(const T* from_ptr, T* to_ptr, size_t elements, StreamView& stream)
    {
        std::copy(from_ptr, from_ptr + elements, to_ptr);
    }
};

#endif


template <typename HardwareUniquePtrFromT, typename HardwareUniquePtrToT>
constexpr void memcpy(const HardwareUniquePtrFromT& from_ptr, HardwareUniquePtrToT& to_ptr, size_t elements)
{
    using FromRemRef = typename std::remove_reference_t<HardwareUniquePtrFromT>;
    using ToRemRef = typename std::remove_reference_t<HardwareUniquePtrToT>;
    using FromT = typename std::remove_const_t<typename FromRemRef::element_type>;
    using T = typename std::remove_const_t<typename ToRemRef::element_type>;
    static_assert(std::is_same_v<FromT, T>, "From and to were not of same type");

    constexpr auto memcpy_type = get_memcpy_type<what_hardware<FromRemRef>(), what_hardware<ToRemRef>()>();
    MemCpyPartialTemplateSpecializer<memcpy_type>::memcpy_data(from_ptr.get(), to_ptr.get(), elements);
}


template <typename HardwareUniquePtrFromT, typename HardwareUniquePtrToT>
constexpr void memcpy_async(const HardwareUniquePtrFromT& from_ptr,
                            HardwareUniquePtrToT& to_ptr,
                            size_t elements,
                            StreamView& stream)
{
    using FromRemRef = typename std::remove_reference_t<HardwareUniquePtrFromT>;
    using ToRemRef = typename std::remove_reference_t<HardwareUniquePtrToT>;
    using FromT = typename std::remove_const_t<typename FromRemRef::element_type>;
    using T = typename std::remove_const_t<typename ToRemRef::element_type>;
    static_assert(std::is_same_v<FromT, T>, "From and to were not of same type");

    constexpr auto memcpy_type = get_memcpy_type<what_hardware<FromRemRef>(), what_hardware<ToRemRef>()>();

    MemCpyPartialTemplateSpecializer<memcpy_type>::memcpy_data_async(from_ptr.get(), to_ptr.get(), elements, stream);
}

template <typename FromT, typename HardwareUniquePtrToT>
constexpr void memcpy(const std::initializer_list<FromT>& from_ptr, HardwareUniquePtrToT& to_ptr, size_t elements)
{
    using ToRemRef = typename std::remove_reference_t<HardwareUniquePtrToT>;
    using T = typename std::remove_const_t<typename ToRemRef::element_type>;
    static_assert(std::is_same_v<FromT, T>, "From and to were not of same type");

    constexpr auto memcpy_type = get_memcpy_type<Hardware::CPU, what_hardware<HardwareUniquePtrToT>()>();
    MemCpyPartialTemplateSpecializer<memcpy_type>::memcpy_data(from_ptr.begin(), to_ptr.get(), elements);
}

// =============== MEMSET ===============

template <Hardware HardwareV>
struct MemsetPartialTemplateSpecializer
{
    template <typename T>
    constexpr static void memset_data(T* ptr, T val, size_t num_bytes);

    template <typename T>
    constexpr static void memset_data_async(T* ptr, T val, size_t num_bytes, StreamView& stream);
};

#ifdef __CUDACC__
template <Hardware HardwareV>
template <typename T>
constexpr void MemsetPartialTemplateSpecializer<HardwareV>::memset_data<T>(T* ptr, T val, size_t num_bytes)
{
    if constexpr (HardwareV == Hardware::CPU)
    {
        std::memset(ptr, val, num_bytes);
    }
    else  // if (hardware == MemcpyType::HostToDevice)
    {
        cudaMemset(ptr, val, num_bytes);
    }
}

template <Hardware HardwareV>
template <typename T>
constexpr void MemsetPartialTemplateSpecializer<HardwareV>::memset_data_async<T>(T* ptr,
                                                                                 T val,
                                                                                 size_t num_bytes,
                                                                                 StreamView& stream)
{
    if constexpr (HardwareV == Hardware::CPU)
    {
        std::memset(ptr, val, num_bytes);
    }
    else  // if (hardware == MemcpyType::HostToDevice)
    {
        cudaMemsetAsync(ptr, val, num_bytes, stream);
    }
}


#else
template <>
struct MemsetPartialTemplateSpecializer<Hardware::CPU>
{
    template <typename T>
    constexpr static void memset_data(T* ptr, T val, size_t num_bytes)
    {
        std::memset(ptr, val, num_bytes);
    }

    template <typename T>
    constexpr static void memset_data_async(T* ptr, T val, size_t num_bytes, StreamView& stream)
    {
        std::memset(ptr, val, num_bytes);
    }
};
#endif


template <typename HardwareUniquePtrT, typename T>
constexpr void memset(HardwareUniquePtrT& ptr, T val, size_t num_bytes)
{
    using RemRef = typename std::remove_reference_t<HardwareUniquePtrT>;
    constexpr auto hardware = what_hardware<RemRef>();

    MemsetPartialTemplateSpecializer<hardware>::memset_data(ptr.get(), val, num_bytes);
}

template <typename HardwareUniquePtrT, typename T>
constexpr void memset_async(HardwareUniquePtrT& ptr, T val, size_t num_bytes, StreamView& stream)
{
    using RemRef = typename std::remove_reference_t<HardwareUniquePtrT>;
    constexpr auto hardware = what_hardware<RemRef>();

    MemsetPartialTemplateSpecializer<hardware>::memset_data_async(ptr.get(), val, num_bytes, stream);
}


}  // namespace cute
