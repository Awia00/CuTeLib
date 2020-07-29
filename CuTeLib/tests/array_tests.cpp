#include <cute/array.h>
#include <doctest/doctest.h>

namespace cute
{

TEST_SUITE("Array")
{
    TEST_CASE("drop")
    {
        const auto i32_arr = Array<int32_t, 3>{ 0, 1, 2 };
        CHECK(i32_arr.drop<0>() == Array<int32_t, 2>{ 1, 2 });
        CHECK(i32_arr.drop<1>() == Array<int32_t, 2>{ 0, 2 });
        CHECK(i32_arr.drop<2>() == Array<int32_t, 2>{ 0, 1 });
    }

    TEST_CASE("take")
    {
        const auto i32_arr = Array<int32_t, 3>{ 0, 1, 2 };
        CHECK(i32_arr.take<1>() == Array<int32_t, 1>{ 0 });
        CHECK(i32_arr.take<2>() == Array<int32_t, 2>{ 0, 1 });
        CHECK(i32_arr.take<3>() == Array<int32_t, 3>{ 0, 1, 2 });
    }

    TEST_CASE("skip")
    {
        const auto i32_arr = Array<int32_t, 3>{ 0, 1, 2 };
        CHECK(i32_arr.skip<0>() == Array<int32_t, 3>{ 0, 1, 2 });
        CHECK(i32_arr.skip<1>() == Array<int32_t, 2>{ 1, 2 });
        CHECK(i32_arr.skip<2>() == Array<int32_t, 1>{ 2 });
    }

    TEST_CASE("mul")
    {
        CHECK(array(1).mul() == 1);
        CHECK(array(1, 10).mul() == 10);
        CHECK(array(1, 10, 100).mul() == 1000);
    }

    TEST_CASE("sum")
    {
        CHECK(array(1).sum() == 1);
        CHECK(array(1, 10).sum() == 11);
        CHECK(array(1, 10, 100).sum() == 111);
    }

    TEST_CASE("sum")
    {
        CHECK(array(1).sum(array(1)) == array(2));
        CHECK(array(1, 10).sum(array(1, 10)) == array(2, 20));
        CHECK(array(1, 10, 100).sum(array(1, 10, 100)) == array(2, 20, 200));
    }

    TEST_CASE("dot_product")
    {
        CHECK(array(1).dot_product(array(1)) == 1);
        CHECK(array(1, 10).dot_product(array(1, 10)) == 101);
        CHECK(array(1, 10, 100).dot_product(array(1, 10, 100)) == 10101);
    }

    TEST_CASE("negate")
    {
        CHECK(array(1).negate() == array(-1));
        CHECK(array(1, 10).negate() == array(-1, -10));
        CHECK(array(1, 10, 100).negate() == array(-1, -10, -100));
    }

    TEST_CASE("cast")
    {
        CHECK(array(1.0).cast<int32_t>() == array(1));
        CHECK(array(1.0, 10.5).cast<int32_t>() == array(1, 10));
        CHECK(array(1.0, 10.5, 100.9999).cast<int32_t>() == array(1, 10, 100));
    }
}

} // namespace cute