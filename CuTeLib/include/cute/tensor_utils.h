#include <cute/array.h>
#include <cute/tensor.h>
#include <numeric>

namespace cute
{

template <typename T, typename ShapesT>
[[nodiscard]] auto iota(ShapesT args, T initial = 0)
{
    auto result = Tensor<T, ShapesT::size(), Hardware::CPU>(std::move(args));
    std::iota(result.begin(), result.end(), initial);
    return result;
}

} // namespace cute