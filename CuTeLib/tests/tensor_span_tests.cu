#include <iostream>
#include "test_utils.h"
#include <cute/array.h>
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
        check_tensors(cpu_transfered, cpu_tensor);  // but it is the same values

        auto gpu_transfered = cpu_transfered.transfer<Hardware::GPU>();
        CHECK(gpu_transfered.data() != cpu_transfered.data());  // its not the same memory

        auto moved_back = gpu_transfered.transfer<Hardware::CPU>();
        check_tensors(moved_back, cpu_tensor);
    }
}

}  // namespace cute
