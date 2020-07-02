#pragma once
#include <cute/defs.h>
#include <ostream>

namespace cute
{

template <typename T, int32_t Length>
class Array
{
    public:
    using value_type = T;
    using size_type = int32_t;

    T data_[Length];

    // static members
    constexpr static bool empty() noexcept
    {
        return Length == 0;
    }
    constexpr static int32_t size() noexcept
    {
        return Length;
    }

    // Members

    /// Returns a pointer to the inner data
    CUTE_DEV_HOST constexpr const T* data() const noexcept
    {
        return this->data_;
    }
    /// Returns a pointer to the inner data
    CUTE_DEV_HOST constexpr T* data() noexcept
    {
        return this->data_;
    }

    /// Returns a const reference to the element at idx position.
    /// No bounds check
    CUTE_DEV_HOST constexpr T& operator[](const int32_t idx) noexcept
    {
        return this->data_[idx];
    }
    /// Returns a reference to the element at idx position.
    /// No bounds check
    CUTE_DEV_HOST constexpr const T& operator[](const int32_t idx) const noexcept
    {
        return this->data_[idx];
    }

    /// Returns the value at idx position in data.
    /// Bounds check runtime.
    CUTE_DEV_HOST constexpr T at(const int32_t idx) const
    {
        assert(idx < Length);
        return this->data_[idx];
    }
    /// Returns a reference to the element at idx position in data.
    /// Bounds check runtime.
    CUTE_DEV_HOST constexpr T& at_ref(const int32_t idx)
    {
        assert(idx < Length);
        return this->data_[idx];
    }
    /// Returns a const reference to the element at idx position in data.
    /// Bounds check runtime.
    CUTE_DEV_HOST constexpr const T& at_ref(const int32_t idx) const
    {
        assert(idx < Length);
        return this->data_[idx];
    }

    /// Returns the value at idx position in data.
    /// Bounds check compile-time.
    template <int32_t Idx>
    CUTE_DEV_HOST constexpr T at() const noexcept
    {
        static_assert(Idx < Length, "Out of bounds");
        return this->data_[Idx];
    }
    /// Returns a reference to the element at idx position in data.
    /// Bounds check compile-time.
    template <int32_t Idx>
    CUTE_DEV_HOST constexpr T& at_ref() noexcept
    {
        static_assert(Idx < Length, "Out of bounds");
        return this->data_[Idx];
    }
    /// Returns a const reference to the element at idx position in data.
    /// Bounds check compile-time.
    template <int32_t Idx>
    CUTE_DEV_HOST constexpr const T& at_ref() const noexcept
    {
        static_assert(Idx < Length, "Out of bounds");
        return this->data_[Idx];
    }


    template <typename AccumulatorT = T>
    CUTE_DEV_HOST constexpr AccumulatorT sum() const noexcept
    {
        auto accum = AccumulatorT(0);
        for (auto i = 0; i < Length; i++)
        {
            accum += this->data_[i];
        }
        return accum;
    }

    CUTE_DEV_HOST constexpr Array<T, Length> sum(const Array<T, Length>& other) const noexcept
    {
        auto res = Array<T, Length>();
        for (auto i = 0; i < Length; i++)
        {
            res[i] = this->data_[i] + other.data_[i];
        }
        return res;
    }

    template <typename AccumulatorT = T>
    CUTE_DEV_HOST constexpr AccumulatorT product() const noexcept
    {
        auto accum = AccumulatorT(1);
        for (auto i = 0; i < Length; i++)
        {
            accum *= this->data_[i];
        }
        return accum;
    }

    template <typename AccumulatorT = T>
    CUTE_DEV_HOST constexpr AccumulatorT inner_product(const Array<T, Length>& other) const noexcept
    {
        auto accum = AccumulatorT(0);
        for (auto i = 0; i < Length; i++)
        {
            accum += this->data_[i] * other.data_[i];
        }
        return accum;
    }

    CUTE_DEV_HOST constexpr Array<T, Length> outer_product(const Array<T, Length>& other) const noexcept
    {
        auto res = Array<T, Length>();
        for (auto i = 0; i < Length; i++)
        {
            res[i] = this->data_[i] * other.data_[i];
        }
        return res;
    }

    template <int32_t SkipN>
    CUTE_DEV_HOST constexpr Array<T, Length - SkipN> skip() const noexcept
    {
        constexpr auto NewLength = Length - SkipN;
        static_assert(NewLength > 0, "Cannot skip more items than available");

        auto res = Array<T, NewLength>();
        for (auto i = SkipN; i < Length; i++)
        {
            res[i - SkipN] = this->data_[i];
        }
        return res;
    }

    template <int32_t TakeN>
    CUTE_DEV_HOST constexpr Array<T, TakeN> take() const noexcept
    {
        static_assert(TakeN < Length, "Cannot take more items than available");

        auto res = Array<T, TakeN>();
        for (auto i = 0; i < TakeN; i++)
        {
            res[i] = this->data_[i];
        }
        return res;
    }

    template <int32_t DropIdx>
    CUTE_DEV_HOST constexpr Array<T, Length - 1> drop() const noexcept
    {
        constexpr auto NewLength = Length - 1;
        static_assert(DropIdx < Length, "Cannot drop index larger than length");

        auto res = Array<T, NewLength>();
        auto res_idx = 0;
        for (auto i = 0; i < DropIdx; i++)
        {
            res[res_idx++] = this->data_[i];
        }
        for (auto i = DropIdx + 1; i < Length; i++)
        {
            res[res_idx++] = this->data_[i];
        }
        return res;
    }
};

template <typename T, typename... Args>
auto array(T first, Args... args)
{
    return Array<T, sizeof...(Args) + 1>{ first, args... };
}

template <typename T, int32_t Length>
auto array(Array<T, Length> arr)
{
    return arr;
}

template <typename T, int32_t Length>
std::ostream& stream_array(std::ostream& stream, const Array<T, Length>& arr, char elem_breaker = ' ')
{
    stream << "[ ";
    if (!arr.empty())
    {
        stream << arr.template at<0>();
    }
    for (auto i = 1; i < Length; i++)
    {
        stream << "," << elem_breaker << arr[i];
    }
    stream << " ]";
    return stream;
}
} // namespace cute
