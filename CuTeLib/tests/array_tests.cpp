#include <sstream>
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
        CHECK_EQ(Array<int32_t, 0>().mul(), 1);
        CHECK_EQ(array(0).mul(), 0);
        CHECK_EQ(array(1).mul(), 1);
        CHECK_EQ(array(2).mul(), 2);
        CHECK_EQ(array(2, 10).mul(), 20);
        CHECK_EQ(array(2, 10, 100).mul(), 2000);
    }

    TEST_CASE("sum")
    {
        CHECK_EQ(Array<int32_t, 0>().sum(), 0);
        CHECK_EQ(array(1).sum(), 1);
        CHECK_EQ(array(1, 10).sum(), 11);
        CHECK_EQ(array(1, 10, 100).sum(), 111);
    }

    TEST_CASE("sum")
    {
        CHECK_EQ(Array<int32_t, 0>().sum(Array<int32_t, 0>()), Array<int32_t, 0>());
        CHECK_EQ(array(1).sum(array(1)), array(2));
        CHECK_EQ(array(1, 10).sum(array(1, 10)), array(2, 20));
        CHECK_EQ(array(1, 10, 100).sum(array(1, 10, 100)), array(2, 20, 200));
    }

    TEST_CASE("dot_product")
    {
        CHECK_EQ(Array<int32_t, 0>().dot_product(Array<int32_t, 0>()), 0);
        CHECK_EQ(array(1).dot_product(array(1)), 1);
        CHECK_EQ(array(1, 10).dot_product(array(1, 10)), 101);
        CHECK_EQ(array(1, 10, 100).dot_product(array(1, 10, 100)), 10101);
    }

    TEST_CASE("negate")
    {
        CHECK_EQ(array(1).negate(), array(-1));
        CHECK_EQ(array(1, 10).negate(), array(-1, -10));
        CHECK_EQ(array(1, 10, 100).negate(), array(-1, -10, -100));

        CHECK_EQ(array(1).negate().negate(), array(1));
        CHECK_EQ(array(1, 10).negate().negate(), array(1, 10));
        CHECK_EQ(array(1, 10, 100).negate().negate(), array(1, 10, 100));
    }

    TEST_CASE("cast")
    {
        CHECK_EQ(Array<double, 0>().cast<int32_t>(), Array<int32_t, 0>());
        CHECK_EQ(array(1.0).cast<int32_t>(), array(1));
        CHECK_EQ(array(1.0, 10.5).cast<int32_t>(), array(1, 10));
        CHECK_EQ(array(1.0, 10.5, 100.9999).cast<int32_t>(), array(1, 10, 100));
    }

    TEST_CASE("operator<<")
    {
        auto to_string = [](const auto& elem) -> std::string
        {
            auto ss = std::stringstream{};
            ss << elem;
            return ss.str();
        };
        CHECK_EQ(to_string(Array<int32_t, 0>()), "[]");
        CHECK_EQ(to_string(array(1)), "[1]");
        CHECK_EQ(to_string(array(1, 10)), "[1, 10]");
        CHECK_EQ(to_string(array(1, 10, 100)), "[1, 10, 100]");
    }
}

}  // namespace cute
