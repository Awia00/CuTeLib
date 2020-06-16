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
};

template <typename T, int32_t RankV, Hardware HardwareV, typename Traits = TensorTraits>
class Tensor
{
    public:
    using shape_type = typename Traits::shape_type;
    using size_type = typename Traits::size_type;
    using index_type = typename Traits::index_type;
    using value_type = T;

    private:
    HardwareUniquePtr<T, HardwareV> data_;
    Array<shape_type, RankV> shape_;

    public:
    explicit Tensor(Array<shape_type, RankV> shape)
      : data_(cute::make_unique<T, HardwareV>(shape.template product<size_type>()))
      , shape_(std::move(shape))
    {
    }

    static constexpr Hardware hardware() noexcept
    {
        return HardwareV;
    }
    static constexpr int32_t rank() noexcept
    {
        return RankV;
    }

    bool empty() const noexcept
    {
        return this->shape_.empty();
    }

    size_t size() const noexcept
    {
        return this->shape_.template product<size_t>();
    }

    shape_type shape(int32_t dim) const noexcept
    {
        return this->shape_[dim];
    }

    template <int32_t Idx>
    shape_type shape() const noexcept
    {
        return this->shape_.template at<Idx>();
    }

    TensorSpan<T, RankV, HardwareV> get_span() noexcept
    {
        return get_span_of_data(this->data_, this->shape_);
    }

    TensorSpan<const T, RankV, HardwareV> get_span() const noexcept
    {
        return get_span_of_data(this->data_, this->shape_);
    }

    operator TensorSpan<T, RankV, HardwareV>() noexcept
    {
        return this->get_span();
    }
    operator TensorSpan<const T, RankV, HardwareV>() const noexcept
    {
        return this->get_span();
    }
};

template <typename T, Hardware HardwareV, typename Traits = TensorTraits>
using Vector = Tensor<T, 1, HardwareV, Traits>;

template <typename T, Hardware HardwareV, typename Traits = TensorTraits>
using Matrix = Tensor<T, 2, HardwareV, Traits>;

template <typename T, Hardware HardwareV, typename Traits = TensorTraits>
using Cube = Tensor<T, 3, HardwareV, Traits>;


} // namespace cute