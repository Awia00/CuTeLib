#pragma once
#include <atomic>
#include <functional>
#include <utility>
#include <cuda/std/atomic>
#include <cuda/std/functional>
#include <cute/array.h>
#include <cute/defs.h>
#include <cute/hardware.h>
#include <cute/tensor.h>
#include <cute/tensor_span.h>

#ifdef _MSC_VER
#include <intrin.h>
#define __builtin_popcount __popcnt
#endif

namespace cute::experimental
{


/**
 * @brief Simple Hash implementation for int32_t - given another type it will just try to do a static_cast<int32_t>.
 *
 * @tparam T
 * @tparam PrecisionV for values < 32 speed is increased for the price of precision
 */
template <typename T, int32_t PrecisionV = 32>
struct Hash
{
    static_assert(PrecisionV <= 32, "exceeded the hash_weights");

    CUTE_DEV_HOST
    constexpr static int32_t hash(int32_t x) noexcept
    {
        constexpr auto hash_weights
            = Array<uint32_t, 32>{ 0x21ae4036, 0x32435171, 0xac3338cf, 0xea97b40c, 0x0e504b22,
                                   0x9ff9a4ef, 0x111d014d, 0x934f3787, 0x6cd079bf, 0x69db5c31,
                                   0xdf3c28ed, 0x40daf2ad, 0x82a5891c, 0x4659c7b0, 0x73dc0ca8,
                                   0xdad3aca2, 0x00c74c7e, 0x9a2521e2, 0xf38eb6aa, 0x64711ab6,
                                   0x5823150a, 0xd13a3a9a, 0x30a5aa04, 0x0fb9a1da, 0xef785119,
                                   0xc9f0b067, 0x1e7dde42, 0xdda4a7b2, 0x1a1c2640, 0x297c0633,
                                   0x744edb48, 0x19adce93 }
                  .take<PrecisionV>();

        auto r = 0;
        for (auto i = 0; i < hash_weights.size(); i++)
        {
#ifdef __CUDA_ARCH__
            r += (__popc(hash_weights[i] & x) & 1) << (hash_weights.size() - i - 1);
#else
            r += (__builtin_popcount(hash_weights[i] & x) & 1) << (hash_weights.size() - i - 1);
#endif
        }
        return r;
    }

    CUTE_DEV_HOST constexpr int32_t operator()(const T& elem) const noexcept
    {
        return this->hash(static_cast<int32_t>(elem));
    }
};


enum struct state_type
{
    empty,
    reserved,
    filled
};

enum struct insert_type
{
    inserted,
    key_existed,
    map_full
};

/**
 * @brief Based on https://youtu.be/-ENnYEWezKo?t=1921
 *
 * @tparam KeyT
 * @tparam ValueT
 * @tparam HardwareV
 * @tparam HashT
 * @tparam EqualT
 */
template <typename KeyT, typename ValueT, Hardware HardwareV, typename HashT = Hash<KeyT>, typename EqualT = cuda::std::equal_to<KeyT>>
struct InsertOnlyMapSpan
{
    using key_type = KeyT;
    using value_type = ValueT;

    uint64_t capacity_;
    TensorSpan<KeyT, 1, HardwareV> keys_;
    TensorSpan<ValueT, 1, HardwareV> values_;
    TensorSpan<cuda::std::atomic<state_type>, 1, HardwareV> states_;
    HashT hash_{};
    EqualT equal_{};

    CUTE_DEV_HOST ValueT* get(const KeyT& key)
    {
        auto index = this->hash_(key) % this->capacity_;
        for (auto i = 0; i < this->capacity_; i++)
        {
            state_type state;  // this->states_[index].wait(state_type::reserved, cuda::std::memory_order_acquire); // TODO(anders.wind): C++20
            do
            {
                state = this->states_[index].load(cuda::std::memory_order_acquire);
            } while (state == state_type::reserved);

            if (state == state_type::filled)
            {
                if (this->equal_(this->keys_[index], key))
                {
                    return this->values_.data() + index;
                }
                index = (index + 1) % this->capacity_;
            }
            else
            {
                return nullptr;
            }
        }
        return nullptr;
    }

    CUTE_DEV_HOST insert_type try_insert(const KeyT& key, const ValueT& value)
    {
        auto index = this->hash_(key) % this->capacity_;
        for (auto i = 0; i < this->capacity_; i++)
        {
            auto old = this->states_[index].load(cuda::std::memory_order_acquire);
            while (old == state_type::empty)
            {
                if (this->states_[index].compare_exchange_weak(old, state_type::reserved, cuda::std::memory_order_acq_rel))
                {
                    this->keys_[index] = key;  // TODO(anders.wind): why did they allocate here?
                    this->values_[index] = value;  // TODO(anders.wind): why did they allocate here?
                    this->states_[index].store(state_type::filled, cuda::std::memory_order_release);
                    // this->states_[index].notify_all(); // TODO(anders.wind): C++20
                    return insert_type::inserted;
                }
            }
            // this->states_[index].wait(state_type::reserved, cuda::std::memory_order_acquire); // TODO(anders.wind): C++20
            while (state_type::reserved == this->states_[index].load(cuda::std::memory_order_acquire))
            {
            }

            if (this->equal_(this->keys_[index], key))
            {
                return insert_type::key_existed;
            }
            index = (index + 1) % this->capacity_;
        }
        return insert_type::map_full;
    }
};

struct AtomicTensorTraits
{
    using shape_type = int32_t;
    using size_type = int64_t;
    using index_type = int32_t;

    constexpr static bool zero_initialize = false;
};

template <typename KeyT, typename ValueT, Hardware HardwareV, typename HashT = Hash<KeyT>, typename EqualT = cuda::std::equal_to<KeyT>>
struct InsertOnlyMap
{
    using MyT = InsertOnlyMap<KeyT, ValueT, HardwareV, HashT, EqualT>;
    using SpanT = InsertOnlyMapSpan<KeyT, ValueT, HardwareV, HashT, EqualT>;

    using key_type = KeyT;
    using value_type = ValueT;

    using state_type = state_type;
    using insert_type = insert_type;

    uint64_t capacity_;
    Tensor<KeyT, 1, HardwareV> keys_;
    Tensor<ValueT, 1, HardwareV> values_;
    Tensor<cuda::std::atomic<state_type>, 1, HardwareV> states_;

    explicit InsertOnlyMap(uint64_t capacity)
      : capacity_(capacity)
      , keys_(shape(static_cast<int32_t>(this->capacity_)))  // TODO(anders.wind): think about conversion
      , values_(shape(static_cast<int32_t>(this->capacity_)))  // TODO(anders.wind): think about conversion
      , states_(shape(static_cast<int32_t>(this->capacity_)))  // TODO(anders.wind): think about conversion
    {
    }

    InsertOnlyMap(uint64_t capacity,
                  Tensor<KeyT, 1, HardwareV> keys,
                  Tensor<ValueT, 1, HardwareV> values,
                  Tensor<cuda::std::atomic<state_type>, 1, HardwareV> states)
      : capacity_(capacity), keys_(std::move(keys)), values_(std::move(values)), states_(std::move(states))
    {
        assert(this->keys_.shape(0) == this->capacity_);
        assert(this->values_.shape(0) == this->capacity_);
        assert(this->states_.shape(0) == this->capacity_);
    }

    SpanT get_span()
    {
        return InsertOnlyMapSpan<KeyT, ValueT, HardwareV, HashT, EqualT>{
            this->capacity_, this->keys_.get_span(), this->values_.get_span(), this->states_.get_span()
        };
    }

    operator SpanT()
    {
        return this->get_span();
    }

    template <Hardware OtherHardwareV>
    InsertOnlyMap<KeyT, ValueT, OtherHardwareV, HashT, EqualT> transfer()
    {
        return InsertOnlyMap<KeyT, ValueT, OtherHardwareV, HashT, EqualT>(
            this->capacity_,
            this->keys_.transfer<OtherHardwareV>(),
            this->values_.transfer<OtherHardwareV>(),
            this->states_.transfer<OtherHardwareV>());
    }
};


}  // namespace cute::experimental
