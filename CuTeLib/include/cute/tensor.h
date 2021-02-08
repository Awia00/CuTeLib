#pragma once
#include <cute/array.h>
#include <cute/defs.h>
#include <cute/tensor_span.h>
#include <cute/unique_ptr.h>

namespace cute
{

struct TensorTraits
{
    using shape_type = int32_t;
    using size_type = int64_t;
    using index_type = int32_t;

    constexpr static bool zero_initialize = true;
};

///
/// A Tensor is a multidimensional data container.
/// To access its elements, call get_span().
/// Tensors are either CPU or GPU, therefore you can create type safe functions and overload which will work for either GPU or CPU functions.
///
template <typename T, int32_t RankV, Hardware HardwareV, typename Traits = TensorTraits>
class Tensor
{
    public:
    using shape_type = typename Traits::shape_type;
    using size_type = typename Traits::size_type;
    using index_type = typename Traits::index_type;
    using value_type = T;

    using MyT = Tensor<T, RankV, HardwareV, Traits>;

    private:
    HardwareUniquePtr<T[], HardwareV> data_;
    Array<shape_type, RankV> shape_;

    public:
    explicit Tensor(Array<shape_type, RankV> shape) noexcept
      : data_(cute::make_unique<T[], HardwareV>(shape.template mul<size_type>())), shape_(std::move(shape))
    {
        static_assert(!std::is_array_v<T>, "T should not be a array type");
        static_assert(!std::is_const_v<T>, "T should not be const");

        if constexpr (Traits::zero_initialize)
        {
            memset(this->data_, T(), this->size() * sizeof(T));
        }
    }

    Tensor(const std::initializer_list<T>& init_list, Array<shape_type, RankV> shape) noexcept
      : data_(cute::make_unique<T[], HardwareV>(shape.template mul<size_type>())), shape_(std::move(shape))
    {
        static_assert(!std::is_array_v<T>, "T should not be a array type");
        static_assert(!std::is_const_v<T>, "T should not be const");
        assert(init_list.size() == shape.template mul<size_type>());

        memcpy(init_list, this->data_, init_list.size());
    }

    // Templated copy constructor
    template <typename OtherTensorT>
    explicit Tensor(const OtherTensorT& other_tensor) noexcept
      : data_(cute::make_unique<T[], HardwareV>(other_tensor.get_shape().template mul<size_type>()))
      , shape_(other_tensor.get_shape())
    {
        static_assert(!std::is_array_v<T>, "T should not be a array type");
        static_assert(!std::is_const_v<T>, "T should not be const");

        memcpy(other_tensor.data_ptr(), this->data_, other_tensor.size());
    }

    ///
    /// Templated copy assignment
    template <typename OtherTensorT>
    MyT& operator=(const OtherTensorT& other_tensor) noexcept
    {
        static_assert(!std::is_array_v<T>, "T should not be a array type");
        static_assert(!std::is_const_v<T>, "T should not be const");

        if (this->size() != other_tensor.size())
        {
            this->data_.reset();
            this->data_ = cute::make_unique<T[], HardwareV>(other_tensor.size());
        }
        this->shape_ = other_tensor.get_shape();

        memcpy(other_tensor.data_ptr(), this->data_, this->size());
        return *this;
    }

    Tensor(const MyT& other_tensor) noexcept
      : data_(cute::make_unique<T[], HardwareV>(other_tensor.get_shape().template mul<size_type>()))
      , shape_(other_tensor.get_shape())
    {
        static_assert(!std::is_array_v<T>, "T should not be a array type");
        static_assert(!std::is_const_v<T>, "T should not be const");

        memcpy(other_tensor.data_ptr(), this->data_, other_tensor.size());
    }
    MyT& operator=(const MyT& other_tensor) noexcept
    {
        static_assert(!std::is_array_v<T>, "T should not be a array type");
        static_assert(!std::is_const_v<T>, "T should not be const");

        if (this->size() != other_tensor.size())
        {
            this->data_.reset();
            this->data_ = cute::make_unique<T[], HardwareV>(other_tensor.size());
        }
        this->shape_ = other_tensor.get_shape();

        memcpy(other_tensor.data_ptr(), this->data_, this->size());
        return *this;
    }

    Tensor(MyT&& other_tensor) noexcept = default;
    MyT& operator=(MyT&& other_tensor) noexcept = default;

    /// Transfer copies this tensor to the specified hardware
    template <Hardware ToHardwareV>
    [[nodiscard]] constexpr auto transfer() const noexcept
    {
        return Tensor<T, RankV, ToHardwareV, Traits>(*this);
    }

    template <Hardware ToHardwareV, typename StreamT>
    Tensor<T, RankV, ToHardwareV> transfer_async(StreamT& stream) const
    {
        auto res = Tensor<T, RankV, ToHardwareV>(this->shape_);
        this->transfer_async<ToHardwareV, StreamT>(stream, res);
        return res;
    }
    template <Hardware ToHardwareV, typename StreamT>
    void transfer_async(StreamT& stream, const TensorSpan<T, RankV, ToHardwareV>& out) const
    {
        assert(this->shape_ == out.get_shape());
        constexpr auto memcpy_type = get_memcpy_type<HardwareV, ToHardwareV>();

        MemCpyPartialTemplateSpecializer<memcpy_type>::template memcpy_data_async<T>(this->data(),
                                                                                     out.data(),
                                                                                     this->size(),
                                                                                     stream);
    }

    static constexpr Hardware hardware() noexcept
    {
        return HardwareV;
    }
    static constexpr int32_t rank() noexcept
    {
        return RankV;
    }

    [[nodiscard]] constexpr bool empty() const noexcept
    {
        return this->size() == 0;
    }

    [[nodiscard]] constexpr size_t size() const noexcept
    {
        return this->shape_.template mul<size_t>();
    }

    [[nodiscard]] constexpr const Array<shape_type, RankV>& get_shape() const noexcept
    {
        return this->shape_;
    }

    [[nodiscard]] constexpr shape_type shape(int32_t dim) const noexcept
    {
        return this->shape_[dim];
    }

    template <int32_t Idx>
    [[nodiscard]] constexpr shape_type shape() const noexcept
    {
        return this->shape_.template at<Idx>();
    }

    [[nodiscard]] constexpr T* data() noexcept
    {
        return this->data_.get();
    }
    [[nodiscard]] constexpr const T* data() const noexcept
    {
        return this->data_.get();
    }

    ///
    /// @warning: raw pointer return, could be a GPU pointer. Express caution.
    [[nodiscard]] constexpr T* begin() noexcept
    {
        return this->data();
    }

    ///
    /// @warning: raw pointer return, could be a GPU pointer. Express caution.
    [[nodiscard]] constexpr T* end() noexcept
    {
        return this->data() + this->size();
    }

    ///
    /// @warning: raw pointer return, could be a GPU pointer. Express caution.
    [[nodiscard]] constexpr const T* begin() const noexcept
    {
        return this->data();
    }

    ///
    /// @warning: raw pointer return, could be a GPU pointer. Express caution.
    [[nodiscard]] constexpr const T* end() const noexcept
    {
        return this->data() + this->size();
    }

    [[nodiscard]] constexpr const HardwareUniquePtr<T[], HardwareV>& data_ptr() const noexcept
    {
        return this->data_;
    }

    [[nodiscard]] constexpr TensorSpan<T, RankV, HardwareV> get_span() noexcept
    {
        return get_span_of(this->data_, this->shape_);
    }

    [[nodiscard]] constexpr TensorSpan<const T, RankV, HardwareV> get_span() const noexcept
    {
        return get_span_of(this->data_, this->shape_);
    }

    [[nodiscard]] constexpr operator TensorSpan<T, RankV, HardwareV>() noexcept
    {
        return this->get_span();
    }
    [[nodiscard]] constexpr operator TensorSpan<const T, RankV, HardwareV>() const noexcept
    {
        return this->get_span();
    }
};

template <typename T, int32_t RankV, typename Traits>
std::ostream& operator<<(std::ostream& stream, const Tensor<T, RankV, Hardware::CPU, Traits>& tensor)
{
    return stream << tensor.get_span();
}


template <typename T, Hardware HardwareV, typename Traits = TensorTraits>
using Vector = Tensor<T, 1, HardwareV, Traits>;

template <typename T, Hardware HardwareV, typename Traits = TensorTraits>
using Matrix = Tensor<T, 2, HardwareV, Traits>;

template <typename T, Hardware HardwareV, typename Traits = TensorTraits>
using Cube = Tensor<T, 3, HardwareV, Traits>;


} // namespace cute
