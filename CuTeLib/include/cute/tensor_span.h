#pragma once
#include <cute/array.h>
#include <cute/defs.h>
#include <cute/hardware.h>
#include <cute/unique_ptr.h>

namespace cute
{


struct TensorSpanTraits
{
    using shape_type = int32_t;
    using size_type = int64_t;
    using index_type = int32_t;
    constexpr static bool is_rescricted()
    {
        return false;
    }
};

namespace
{

///
/// The TensorSpanBase is here to provide common functionality for Rank 1 and higher. No reason to
/// use this class for outside of cute - use the Specific TensorSpans below instead
///
template <typename T, int32_t RankV, Hardware HardwareV, typename Traits = TensorSpanTraits>
class [[nodiscard]] TensorSpanBase
{
    public:
    using shape_type = typename Traits::shape_type;
    using size_type = typename Traits::size_type;
    using index_type = typename Traits::index_type;
    using value_type = typename T;
    using ptr_type = typename restricted_ptr<T, Traits::is_rescricted()>;

    protected:
    ptr_type data_;
    Array<shape_type, RankV> shape_;

    // Constructor is protected so you dont create it.
    CUTE_DEV_HOST constexpr TensorSpanBase(T* data, Array<shape_type, RankV> shape)
      : data_(data), shape_(std::move(shape))
    {
    }

    public:
    static constexpr Hardware hardware() noexcept
    {
        return HardwareV;
    }
    static constexpr int32_t rank() noexcept
    {
        return RankV;
    }

    CUTE_DEV_HOST [[nodiscard]] constexpr ptr_type data() const noexcept
    {
        return this->data_;
    }

    CUTE_DEV_HOST [[nodiscard]] constexpr bool empty() const noexcept
    {
        return this->shape_.empty();
    }

    CUTE_DEV_HOST [[nodiscard]] constexpr size_type size() const noexcept
    {
        return this->shape_.template mul<size_type>();
    }

    CUTE_DEV_HOST [[nodiscard]] constexpr const Array<shape_type, RankV>& get_shape() const noexcept
    {
        return this->shape_;
    }

    CUTE_DEV_HOST [[nodiscard]] constexpr shape_type shape(int32_t dim) const noexcept
    {
        return this->shape_[dim];
    }

    template <int32_t Idx>
    CUTE_DEV_HOST [[nodiscard]] constexpr shape_type shape() const noexcept
    {
        return this->shape_.template at<Idx>();
    }

    template <int32_t DimV>
    CUTE_DEV_HOST [[nodiscard]] constexpr index_type stride() const noexcept
    {
        auto stride = 1;
        for (auto i = DimV + 1; i < RankV; i++)
        {
            stride *= this->shape_[i];
        }
        return stride;
    }

    protected:
    template <int32_t DimV>
    CUTE_DEV_HOST [[nodiscard]] constexpr index_type index(index_type index_sum) const noexcept
    {
        static_assert(DimV == RankV, "This case should only happen when DimV is RankV-1");
        return index_sum;
    }

    template <int32_t DimV, typename... Args>
    CUTE_DEV_HOST [[nodiscard]] constexpr index_type index(index_type index_sum, index_type index, Args... args) const noexcept
    {
        static_assert(DimV < RankV, "DimV was more than RankV-1");
        static_assert(DimV >= 0, "DimV was less than 0");

        // For RankV == 1, stride should be optimized away by the compiler since it is 1
        return this->index<DimV + 1>(index_sum * this->shape<DimV>() + index, args...);
    }
};

} // namespace


template <typename T, int32_t RankV, Hardware HardwareV, typename Traits = TensorSpanTraits>
class [[nodiscard]] TensorSpan final : public TensorSpanBase<T, RankV, HardwareV, Traits>
{
    public:
    using SuperT = TensorSpanBase<T, RankV, HardwareV, Traits>;
    using shape_type = typename Traits::shape_type;
    using size_type = typename Traits::size_type;
    using index_type = typename Traits::index_type;
    using value_type = typename T;

    CUTE_DEV_HOST constexpr TensorSpan(T* data, Array<shape_type, RankV> shape)
      : SuperT(data, std::move(shape))
    {
    }

    template <typename... Args>
    CUTE_DEV_HOST [[nodiscard]] constexpr T elem(Args... args) const noexcept
    {
        static_assert(sizeof...(Args) == RankV, "One argument per dimension");

        return this->data_[this->index<0>(0, args...)];
    }

    template <typename... Args>
    CUTE_DEV_HOST constexpr T& elem_ref(Args... args) const noexcept
    {
        static_assert(sizeof...(Args) == RankV, "One argument per dimension");

        return this->data_[this->index<0>(0, args...)];
    }

    CUTE_DEV_HOST [[nodiscard]] constexpr TensorSpan<T, RankV - 1, HardwareV, Traits> operator[](index_type idx) const noexcept
    {
        auto offset = this->stride<0>() * idx;
        auto data_ptr = this->data_ + offset;
        auto new_shape = this->shape_.template skip<1>();
        return TensorSpan<T, RankV - 1, HardwareV, Traits>(data_ptr, std::move(new_shape));
    }

    CUTE_DEV_HOST [[nodiscard]] constexpr auto to_const() const noexcept
    {
        return TensorSpan<const T, RankV, HardwareV, Traits>(this->data_, this->shape_);
    }
};

template <typename T, Hardware HardwareV, typename Traits>
class [[nodiscard]] TensorSpan<T, 1, HardwareV, Traits> final : public TensorSpanBase<T, 1, HardwareV, Traits>
{
    public:
    using SuperT = TensorSpanBase<T, 1, HardwareV, Traits>;
    using shape_type = typename Traits::shape_type;
    using size_type = typename Traits::size_type;
    using index_type = typename Traits::index_type;
    using value_type = typename T;

    CUTE_DEV_HOST constexpr TensorSpan(T * data, Array<shape_type, 1> shape)
      : SuperT(data, std::move(shape))
    {
    }

    template <typename... Args>
    CUTE_DEV_HOST [[nodiscard]] constexpr T elem(Args... args) const noexcept
    {
        static_assert(sizeof...(Args) == 1, "One argument per dimension");

        return this->data_[this->index<0>(0, args...)];
    }

    template <typename... Args>
    CUTE_DEV_HOST constexpr T& elem_ref(Args... args) const noexcept
    {
        static_assert(sizeof...(Args) == 1, "One argument per dimension");

        return this->data_[this->index<0>(0, args...)];
    }

    CUTE_DEV_HOST constexpr T& operator[](index_type idx) const noexcept
    {
        return this->elem_ref(idx);
    }

    CUTE_DEV_HOST [[nodiscard]] constexpr TensorSpan<const T, 1, HardwareV, Traits> to_const() const noexcept
    {
        return TensorSpan<const T, 1, HardwareV, Traits>(this->data_, this->shape_);
    }
};


template <typename T, Hardware HardwareV, typename Traits = TensorSpanTraits>
using VectorSpan = TensorSpan<T, 1, HardwareV, Traits>;

template <typename T, Hardware HardwareV, typename Traits = TensorSpanTraits>
using MatrixSpan = TensorSpan<T, 2, HardwareV, Traits>;

template <typename T, Hardware HardwareV, typename Traits = TensorSpanTraits>
using CubeSpan = TensorSpan<T, 3, HardwareV, Traits>;


template <typename HardwareUniquePtrT, typename ShapeContainerT, typename Traits = TensorSpanTraits>
[[nodiscard]] constexpr auto get_span_of(HardwareUniquePtrT& data, ShapeContainerT&& shapes)
{
    using T = std::remove_pointer_t<decltype(data.get())>;
    using ShapeContainerFixed = typename std::remove_reference_t<ShapeContainerT>;
    constexpr auto RankV = ShapeContainerFixed::size();
    constexpr auto HardwareV = what_hardware<decltype(data)>();
    return TensorSpan<T, RankV, HardwareV, Traits>(data.get(), std::forward<ShapeContainerT>(shapes));
}

template <typename HardwareUniquePtrT, typename ShapeContainerT, typename Traits = TensorSpanTraits>
[[nodiscard]] constexpr auto get_span_of(const HardwareUniquePtrT& data, ShapeContainerT&& shapes)
{
    using T = std::remove_pointer_t<decltype(data.get())>;
    using ShapeContainerFixed = typename std::remove_reference_t<ShapeContainerT>;
    constexpr auto RankV = ShapeContainerFixed::size();
    constexpr auto HardwareV = what_hardware<decltype(data)>();
    return TensorSpan<const T, RankV, HardwareV, Traits>(data.get(), std::forward<ShapeContainerT>(shapes));
}

template <typename... Args>
[[nodiscard]] constexpr auto shape(Args... args)
{
    return Array<TensorSpanTraits::shape_type, sizeof...(Args)>{ args... };
}

} // namespace cute