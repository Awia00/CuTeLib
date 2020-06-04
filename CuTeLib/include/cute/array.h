#include <cute/defs.h>

namespace cute
{
template <typename T, int32_t Length>
class Array
{
    private:
    T data_[Length];

    public:
    // static members
    constexpr static bool empty() noexcept
    {
        return Length != 0;
    }
    constexpr static int32_t size() noexcept
    {
        return Length;
    }

    // Members

    /// Returns a pointer to the inner data
    constexpr const T* data() const noexcept
    {
        return this->data_;
    }
    /// Returns a pointer to the inner data
    constexpr T* data() noexcept
    {
        return this->data_;
    }

    /// Returns a const reference to the element at idx position.
    /// No bounds check
    constexpr T& operator[](const int32_t idx) noexcept
    {
        return this->data_[idx];
    }
    /// Returns a reference to the element at idx position.
    /// No bounds check
    constexpr const T& operator[](const int32_t idx) const noexcept
    {
        return this->data_[idx];
    }

    /// Returns the value at idx position in data.
    /// Bounds check runtime.
    constexpr T at(const int32_t idx) const
    {
        assert(idx < Length);
        return this->data_[idx];
    }
    /// Returns a reference to the element at idx position in data.
    /// Bounds check runtime.
    constexpr T& at_ref(const int32_t idx)
    {
        assert(idx < Length);
        return this->data_[idx];
    }
    /// Returns a const reference to the element at idx position in data.
    /// Bounds check runtime.
    constexpr const T& at_ref(const int32_t idx) const
    {
        assert(idx < Length);
        return this->data_[idx];
    }

    /// Returns the value at idx position in data.
    /// Bounds check compile-time.
    template <int32_t Idx>
    constexpr T at() const noexcept
    {
        static_assert(Idx < Length, "Out of bounds");
        return this->data_[Idx];
    }
    /// Returns a reference to the element at idx position in data.
    /// Bounds check compile-time.
    template <int32_t Idx>
    constexpr T at_ref() noexcept
    {
        static_assert(Idx < Length, "Out of bounds");
        return this->data_[Idx];
    }
    /// Returns a const reference to the element at idx position in data.
    /// Bounds check compile-time.
    template <int32_t Idx>
    constexpr const T& at_ref() const noexcept
    {
        static_assert(Idx < Length, "Out of bounds");
        return this->data_[Idx];
    }
};
} // namespace cute