#include <cuda.h>
#include <cute/map.h>
#include <cute/tensor.h>
#include <cute/tensor_generators.h>
#include <doctest/doctest.h>

namespace cute
{

__global__ void hash_kernel(TensorSpan<const int32_t, 1, Hardware::GPU> from,
                            TensorSpan<int32_t, 1, Hardware::GPU> x)
{
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < from.shape(0))
    {
        auto elem = from[idx];
        auto index = experimental::Hash<int32_t>()(elem) % x.shape(0);
        x[index] = elem;
    }
}

__global__ void map_kernel(TensorSpan<const int32_t, 1, Hardware::GPU> from,
                           experimental::InsertOnlyMapSpan<int32_t, int32_t, Hardware::GPU> x)
{
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < from.shape(0))
    {
        x.try_insert(from[idx], from[idx]);
    }
}


TEST_SUITE("hashing")
{
    TEST_CASE("hash in kernel")
    {
        auto from = random<int32_t>(shape(128)).transfer<Hardware::GPU>();
        auto to = Tensor<int32_t, 1, Hardware::GPU>(shape(32));

        hash_kernel<<<1, 128>>>(from, to);

        std::cout << to.transfer<Hardware::CPU>() << std::endl;
    }


    TEST_CASE("trying the static_size_map CPU")
    {
        using map_t = experimental::InsertOnlyMap<int32_t, int32_t, Hardware::CPU>;
        auto map_owner = map_t(32);
        auto map = map_owner.get_span();

        for (auto i = 0; i < map.capacity_; i++)
        {
            INFO(i);
            CHECK(map.get(i) == nullptr);
            CHECK(map.try_insert(i, i) == map_t::insert_type::inserted);
            CHECK(map.try_insert(i, i) == map_t::insert_type::key_existed);
            CHECK(*map.get(i) == i);
        }
        CHECK(map.try_insert(100, 100) == map_t::insert_type::map_full);
        CHECK(map.get(100) == nullptr);
    }

    TEST_CASE("trying the static_size_map GPU")
    {
        using map_t = experimental::InsertOnlyMap<int32_t, int32_t, Hardware::GPU>;
        auto capacity = 1024;
        auto map_owner = map_t(capacity);

        auto data = iota<int32_t>(shape(capacity)).transfer<Hardware::GPU>();
        map_kernel<<<capacity / 64, 64>>>(data, map_owner);
        auto map_owner_cpu = map_owner.transfer<Hardware::CPU>();
        auto map_cpu = map_owner_cpu.get_span();

        for (auto i = 0; i < capacity; i++)
        {
            INFO(i);
            CHECK(map_cpu.try_insert(i, i) == map_t::insert_type::key_existed);
            CHECK(*map_cpu.get(i) == i);
        }
        CHECK(map_cpu.try_insert(capacity + 1, capacity + 1) == map_t::insert_type::map_full);
        CHECK(map_cpu.get(capacity + 1) == nullptr);
    }
}

}  // namespace cute
