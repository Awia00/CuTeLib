#include <sstream>
#include "test_utils.h"
#include <cute/array.h>
#include <cute/stream.h>
#include <cute/tensor.h>
#include <cute/tensor_generators.h>
#include <cute/tensor_span.h>
#include <cute/unique_ptr.h>
#include <doctest/doctest.h>


namespace cute
{

TEST_SUITE("tensor_span")
{
    TEST_CASE("elem 1d")
    {
        const auto iota_tensor = iota<int32_t>(shape(128));
        auto span = iota_tensor.get_span();
        for (auto i = 0; i < iota_tensor.shape(0); i++)
        {
            CHECK(span.elem(i) == i);
        }
    }

    TEST_CASE("elem 2d")
    {
        const auto iota_tensor = iota<int32_t>(shape(32, 4));
        auto span = iota_tensor.get_span();
        auto counter = 0;
        for (auto i = 0; i < iota_tensor.shape(0); i++)
        {
            for (auto j = 0; j < iota_tensor.shape(1); j++)
            {
                CHECK(span.elem(i, j) == counter++);
            }
        }
    }

    TEST_CASE("elem 3d")
    {
        const auto iota_tensor = iota<int32_t>(shape(2, 16, 4));
        auto span = iota_tensor.get_span();
        auto counter = 0;
        for (auto i = 0; i < iota_tensor.shape(0); i++)
        {
            for (auto j = 0; j < iota_tensor.shape(1); j++)
            {
                for (auto k = 0; k < iota_tensor.shape(2); k++)
                {
                    CHECK(span.elem(i, j, k) == counter++);
                }
            }
        }
    }

    TEST_CASE("elem_ref 1d")
    {
        const auto iota_tensor = iota<int32_t>(shape(128));
        auto span = iota_tensor.get_span();
        for (auto i = 0; i < iota_tensor.shape(0); i++)
        {
            CHECK(span.elem_ref(i) == i);
        }
    }

    TEST_CASE("elem_ref 2d")
    {
        const auto iota_tensor = iota<int32_t>(shape(32, 4));
        auto span = iota_tensor.get_span();
        auto counter = 0;
        for (auto i = 0; i < iota_tensor.shape(0); i++)
        {
            for (auto j = 0; j < iota_tensor.shape(1); j++)
            {
                CHECK(span.elem_ref(i, j) == counter++);
            }
        }
    }

    TEST_CASE("elem_ref 3d")
    {
        const auto iota_tensor = iota<int32_t>(shape(2, 16, 4));
        auto span = iota_tensor.get_span();
        auto counter = 0;
        for (auto i = 0; i < iota_tensor.shape(0); i++)
        {
            for (auto j = 0; j < iota_tensor.shape(1); j++)
            {
                for (auto k = 0; k < iota_tensor.shape(2); k++)
                {
                    CHECK(span.elem_ref(i, j, k) == counter++);
                }
            }
        }
    }

    TEST_CASE("operator[] 1d")
    {
        const auto iota_tensor = iota<int32_t>(shape(128));
        auto span = iota_tensor.get_span();
        for (auto i = 0; i < iota_tensor.shape(0); i++)
        {
            CHECK(span[i] == i);
        }
    }

    TEST_CASE("operator[] 2d")
    {
        const auto iota_tensor = iota<int32_t>(shape(32, 4));
        auto span = iota_tensor.get_span();
        auto counter = 0;
        for (auto i = 0; i < iota_tensor.shape(0); i++)
        {
            auto span_i = span[i];  // 1d span
            for (auto j = 0; j < iota_tensor.shape(1); j++)
            {
                CHECK(span_i[j] == counter++);
            }
        }
    }

    TEST_CASE("operator[] 3d")
    {
        const auto iota_tensor = iota<int32_t>(shape(2, 16, 4));
        auto span = iota_tensor.get_span();
        auto counter = 0;
        for (auto i = 0; i < iota_tensor.shape(0); i++)
        {
            auto span_i = span[i];  // 2d span
            for (auto j = 0; j < iota_tensor.shape(1); j++)
            {
                auto span_i_j = span_i[j];  // 1d span
                for (auto k = 0; k < iota_tensor.shape(2); k++)
                {
                    CHECK(span_i_j[k] == counter++);
                }
            }
        }
    }

    TEST_CASE("transfer")
    {
        const auto cpu_tensor = iota<int32_t>(shape(2, 2));
        auto cpu_transfered = cpu_tensor.transfer<Hardware::CPU>();  // just a regular copy

        CHECK(cpu_tensor.data() != cpu_transfered.data());  // its not the same memory
        CHECK(cute::equal(cpu_transfered, cpu_tensor));  // but it is the same values

        auto gpu_transfered = cpu_transfered.transfer<Hardware::GPU>();
        CHECK(gpu_transfered.data() != cpu_transfered.data());  // its not the same memory

        auto moved_back = gpu_transfered.transfer<Hardware::CPU>();
        CHECK(cute::equal(moved_back, cpu_tensor));
    }

    TEST_CASE("transfer_async")
    {
        const auto cpu_tensor = iota<int32_t>(shape(2, 2));
        auto stream = Stream();
        auto cpu_transfered = cpu_tensor.transfer_async<Hardware::CPU>(stream);  // just a regular copy

        CHECK(cpu_tensor.data() != cpu_transfered.data());  // its not the same memory
        CHECK(cute::equal(cpu_transfered, cpu_tensor));  // but it is the same values

        auto gpu_transfered = cpu_transfered.transfer_async<Hardware::GPU>(stream);
        CHECK(gpu_transfered.data() != cpu_transfered.data());  // its not the same memory

        auto moved_back = gpu_transfered.transfer<Hardware::CPU>();
        stream.synchronize();
        CHECK(cute::equal(moved_back, cpu_tensor));
    }

    TEST_CASE("copy_async")
    {
        const auto cpu_tensor = iota<int32_t>(shape(2, 2));
        auto stream = Stream();
        auto cpu_tensor2 = Tensor<int32_t, 2, Hardware::CPU>(shape(2, 2));

        CHECK(!cute::equal(cpu_tensor2, cpu_tensor));  // but it is the same values
        copy_async(cpu_tensor, cpu_tensor2, stream);

        CHECK(cpu_tensor.data() != cpu_tensor2.data());  // its not the same memory
        CHECK(cute::equal(cpu_tensor2, cpu_tensor));  // but it is the same values

        auto gpu_tensor = Tensor<int32_t, 2, Hardware::GPU>(shape(2, 2));
        copy_async(cpu_tensor, gpu_tensor, stream);
        CHECK(gpu_tensor.data() != cpu_tensor.data());  // its not the same memory

        auto moved_back = Tensor<int32_t, 2, Hardware::CPU>(shape(2, 2));
        copy_async(gpu_tensor, moved_back, stream);
        stream.synchronize();
        CHECK(cute::equal(moved_back, cpu_tensor));
    }

    TEST_CASE("operator<<")
    {
        auto to_string = [](const auto& elem) -> std::string
        {
            auto ss = std::stringstream{};
            ss << elem;
            return ss.str();
        };

        CHECK_EQ(to_string(iota<int32_t>(shape(0))), "[]");
        CHECK_EQ(to_string(iota<int32_t>(shape(1))), "[0]");
        CHECK_EQ(to_string(iota<int32_t>(shape(2))), "[0, 1]");
        CHECK_EQ(to_string(iota<int32_t>(shape(3))), "[0, 1, 2]");
        CHECK_EQ(to_string(iota<int32_t>(shape(0, 0))), "[\n]");
        CHECK_EQ(to_string(iota<int32_t>(shape(1, 1))), "[\n  [0]\n]");
        CHECK_EQ(to_string(iota<int32_t>(shape(1, 2))), "[\n  [0, 1]\n]");
        CHECK_EQ(to_string(iota<int32_t>(shape(2, 1))), "[\n  [0],\n  [1]\n]");
        CHECK_EQ(to_string(iota<int32_t>(shape(2, 2))), "[\n  [0, 1],\n  [2, 3]\n]");
    }
}

}  // namespace cute
