cmake_minimum_required(VERSION 3.18)
project(CuTe VERSION 0.0.1)

option(CUTELIB_BUILD_TESTS "Specifies whether or not to build tests" OFF)
option(CUTELIB_BUILD_EXAMPLES "Specifies whether or not to build examples" OFF)

find_package(CUDAToolkit)

add_subdirectory(third_party)
add_subdirectory(CuTeLib)

