cmake_minimum_required(VERSION 3.18)
project(CuTe VERSION 0.0.1)

option(CUTELIB_EXCEPTIONS "Specifies whether or not to error check" ON)
option(CUTELIB_BUILD_TESTS "Specifies whether or not to build tests" OFF)
option(CUTELIB_BUILD_EXAMPLES "Specifies whether or not to build examples" OFF)

include(cmake/CPM.cmake)

# dependencies
find_package(CUDAToolkit)
if(CUTELIB_BUILD_TESTS)
    CPMAddPackage("gh:onqtam/doctest#2.4.5")
endif()

# project code
add_subdirectory(CuTeLib)

