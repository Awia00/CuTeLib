#pragma once

#include <sstream>
#include <string_view>
#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

namespace cute
{

template <typename CudaErrorT>
void cuda_assert(const CudaErrorT& error, std::string_view file, int line);

template <typename CudaErrorT>
void cuda_notify(const CudaErrorT& error, std::string_view file, int line);

#ifdef __CUDACC__

template <typename CudaErrorT>
void cuda_assert(const CudaErrorT& error, std::string_view file, int line)
{
    if (error != cudaSuccess)
    {
        auto ss = std::stringstream();
        ss << file << "[" << line << "]: " << cudaGetErrorString(error);
        throw std::runtime_error(ss.str());
    }
}

template <typename CudaErrorT>
void cuda_notify(const CudaErrorT& error, std::string_view file, int line)
{
    if (error != cudaSuccess)
    {
        auto ss = std::stringstream();
        ss << file << "[" << line << "]: " << cudaGetErrorString(error);
        std::cerr << ss.str() << std::endl;
    }
}
#endif

}  // namespace cute

#ifdef CUTELIB_EXCEPTIONS

#ifndef CUTE_NOEXCEPT  // You can define your own
#define CUTE_NOEXCEPT false
#endif

#ifndef CUTE_ERROR_CHECK  // You can define your own
#define CUTE_ERROR_CHECK(expression) cuda_assert((expression), __FILE__, __LINE__)
#define CUTE_ERROR_NOTIFY(expression) cuda_notify((expression), __FILE__, __LINE__)
#endif

#ifndef CUTE_LAST_ERROR_CHECK  // You can define your own
#define CUTE_LAST_ERROR_CHECK() cuda_assert(cudaGetLastError(), __FILE__, __LINE__)
#define CUTE_LAST_ERROR_NOTIFY(expression) cuda_notify(cudaGetLastError(), __FILE__, __LINE__)
#endif

#else

#define CUTE_NOEXCEPT true
#define CUTE_ERROR_CHECK(expression) (expression)
#define CUTE_LAST_ERROR_CHECK()

#endif
