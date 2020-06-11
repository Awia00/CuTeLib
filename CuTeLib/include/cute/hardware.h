#pragma once
#include <cute/defs.h>

namespace cute
{

enum struct Hardware
{
    CPU,
    GPU
};

// You are free to define your own shorthands for these.
// Maybe a namespace?
//
// namespace HW
// {
// constexpr auto CPU = cute::Hardware::CPU;
// constexpr auto GPU = cute::Hardware::GPU;
// } // namespace HW
//
// to enable the syntax: HW::CPU, HW::GPU

} // namespace cute