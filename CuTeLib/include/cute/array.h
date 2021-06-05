#pragma once
#include <ostream>
#include <utility>
#include <cute/defs.h>

namespace cute
{
struct ArrayTraits
{
    using size_type = int32_t;
    using index_type = int32_t;
};

template <typename T, int32_t Length, typename Traits = ArrayTraits>
struct Array
{
    using value_type = T;
    using size_type = typename ArrayTraits::size_type;
    using index_type = typename ArrayTraits::index_type;

    T data_[Length];


    // ======= Static constexpr Members =======

    ///
    /// Returns true if length is 0 or below
    CUTE_DEV_HOST constexpr static bool empty() noexcept
    {
        return Length <= 0;
    }

    ///
    /// Returns the size of the array - that is the Length template
    CUTE_DEV_HOST constexpr static size_type size() noexcept
    {
        return Length;
    }


    // ======= Members =======

    ///
    /// Returns a pointer to the inner data
    CUTE_DEV_HOST [[nodiscard]] constexpr const T* data() const noexcept
    {
        return this->data_;
    }

    ///
    /// Returns a pointer to the inner data
    CUTE_DEV_HOST [[nodiscard]] constexpr T* data() noexcept
    {
        return this->data_;
    }

    ///
    /// Returns a const reference to the element at idx position.
    /// No bounds check
    CUTE_DEV_HOST [[nodiscard]] constexpr T& operator[](const index_type idx) noexcept
    {
        return this->data_[idx];
    }

    ///
    /// Returns a reference to the element at idx position.
    /// No bounds check
    CUTE_DEV_HOST [[nodiscard]] constexpr const T& operator[](const index_type idx) const noexcept
    {
        return this->data_[idx];
    }


    ///
    /// Returns a ref to the value at idx position in data.
    /// Bounds check runtime.
    CUTE_DEV_HOST [[nodiscard]] constexpr T& at(const index_type idx)
    {
        assert(idx < Length);
        return this->data_[idx];
    }

    ///
    /// Returns a ref to the value at idx position in data.
    /// Bounds check runtime.
    CUTE_DEV_HOST [[nodiscard]] constexpr const T& at(const index_type idx) const
    {
        assert(idx < Length);
        return this->data_[idx];
    }


    ///
    /// Returns a ref to the value at idx position in data.
    /// Bounds check compile-time.
    template <int32_t Idx>
    CUTE_DEV_HOST [[nodiscard]] constexpr T& at() noexcept
    {
        static_assert(Idx < Length, "Out of bounds");
        return this->data_[Idx];
    }

    ///
    /// Returns a ref to the value at idx position in data.
    /// Bounds check compile-time.
    template <int32_t Idx>
    CUTE_DEV_HOST [[nodiscard]] constexpr const T& at() const noexcept
    {
        static_assert(Idx < Length, "Out of bounds");
        return this->data_[Idx];
    }

    ///
    /// Return true if all the elements this and other are equal
    CUTE_DEV_HOST [[nodiscard]] constexpr bool operator==(const Array<T, Length>& other) const noexcept
    {
        auto res = true;
        for (auto i = 0; i < Length; i++)
        {
            res &= this->data_[i] == other.data_[i];
        }
        return res;
    }

    ///
    /// Return true if one of the elements of this and other are not equal
    CUTE_DEV_HOST [[nodiscard]] constexpr bool operator!=(const Array<T, Length>& other) const noexcept
    {
        auto res = true;
        for (auto i = 0; i < Length; i++)
        {
            res &= this->data_[i] == other.data_[i];
        }
        return !res;
    }

    ///
    /// Compute the sum of the elements of this
    template <typename AccumulatorT = T>
    CUTE_DEV_HOST [[nodiscard]] constexpr AccumulatorT sum() const noexcept
    {
        auto accum = AccumulatorT(0);
        for (auto i = 0; i < Length; i++)
        {
            accum += this->data_[i];
        }
        return accum;
    }

    ///
    /// Computes the elementwise summation of this and other.
    CUTE_DEV_HOST [[nodiscard]] constexpr Array<T, Length> sum(const Array<T, Length>& other) const noexcept
    {
        auto res = Array<T, Length>();
        for (auto i = 0; i < Length; i++)
        {
            res[i] = this->data_[i] + other.data_[i];
        }
        return res;
    }

    ///
    /// Negates each elements of this.
    CUTE_DEV_HOST [[nodiscard]] constexpr Array<T, Length> negate() const noexcept
    {
        auto res = Array<T, Length>();
        for (auto i = 0; i < Length; i++)
        {
            res[i] = -this->data_[i];
        }
        return res;
    }

    ///
    /// Casts the elements of the array to the templated type OtherT
    template <typename OtherT>
    CUTE_DEV_HOST [[nodiscard]] constexpr Array<OtherT, Length> cast() const noexcept
    {
        auto res = Array<OtherT, Length>();
        for (auto i = 0; i < Length; i++)
        {
            res[i] = static_cast<OtherT>(this->data_[i]);
        }
        return res;
    }

    ///
    /// Multiplies each elements of this with the input scalar
    CUTE_DEV_HOST [[nodiscard]] constexpr Array<T, Length> product(const T& scalar) const noexcept
    {
        auto res = Array<T, Length>();
        for (auto i = 0; i < Length; i++)
        {
            res[i] = this->data_[i] * scalar;
        }
        return res;
    }

    ///
    /// The elementwise product of the elements of this and other
    CUTE_DEV_HOST [[nodiscard]] constexpr Array<T, Length> product(const Array<T, Length>& other) const noexcept
    {
        auto res = Array<T, Length>();
        for (auto i = 0; i < Length; i++)
        {
            res[i] = this->data_[i] * other.data_[i];
        }
        return res;
    }

    ///
    /// Multiplies the elements of the array with each other (return 1 * elem[0] * elem[1]...)
    template <typename AccumulatorT = T>
    CUTE_DEV_HOST [[nodiscard]] constexpr AccumulatorT mul() const noexcept
    {
        auto accum = AccumulatorT(1);
        for (auto i = 0; i < Length; i++)
        {
            accum *= this->data_[i];
        }
        return accum;
    }

    ///
    /// Compute the dot product between this and other arrays.
    template <typename AccumulatorT = T>
    CUTE_DEV_HOST [[nodiscard]] constexpr AccumulatorT dot_product(const Array<T, Length>& other) const noexcept
    {
        auto accum = AccumulatorT(0);
        for (auto i = 0; i < Length; i++)
        {
            accum += this->data_[i] * other.data_[i];
        }
        return accum;
    }

    ///
    /// Skip the first N elements of the array, indicated by the template parameter SkipN
    template <int32_t SkipN>
    CUTE_DEV_HOST [[nodiscard]] constexpr Array<T, Length - SkipN> skip() const noexcept
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

    ///
    /// Take the first N elements of the array, indicated by the template parameter TakeN
    template <int32_t TakeN>
    CUTE_DEV_HOST [[nodiscard]] constexpr Array<T, TakeN> take() const noexcept
    {
        static_assert(TakeN <= Length, "Cannot take more items than available");

        auto res = Array<T, TakeN>();
        for (auto i = 0; i < TakeN; i++)
        {
            res[i] = this->data_[i];
        }
        return res;
    }

    ///
    /// Drops the element at the templated index DropIdx
    template <int32_t DropIdx>
    CUTE_DEV_HOST [[nodiscard]] constexpr Array<T, Length - 1> drop() const noexcept
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

    template <typename _T, int32_t _Length, typename _Traits>
    friend std::ostream& operator<<(std::ostream&, const Array<_T, _Length, _Traits>& arr);
};

///
/// Variadic template for creating an array instead of relying on list initialization
template <typename T, typename... Args>
[[nodiscard]] constexpr auto array(T first, Args... args)
{
    return Array<T, sizeof...(Args) + 1>{ std::forward<T>(first), std::forward<Args>(args)... };
}

template <typename T, int32_t Length, typename Traits>
std::ostream& operator<<(std::ostream& stream, const Array<T, Length, Traits>& arr)
{
    stream << "[";
    if (!arr.empty())
    {
        stream << arr.template at<0>();
    }
    for (auto i = 1; i < Length; i++)
    {
        stream << ", " << arr[i];
    }
    stream << "]";
    return stream;
}

#ifdef __CUDACC__

static inline dim3 as_dim3(const Array<int32_t, 1>& arr, int32_t y = 1, int32_t z = 1)
{
    return dim3(arr[0], y, z);
}

static inline dim3 as_dim3(const Array<int32_t, 2>& arr, int32_t z = 1)
{
    return dim3(arr[0], arr[1], z);
}

static inline dim3 as_dim3(const Array<int32_t, 3>& arr)
{
    return dim3(arr[0], arr[1], arr[2]);
}

#endif

}  // namespace cute
