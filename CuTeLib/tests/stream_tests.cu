
#include <cute/graph.h>
#include <cute/stream.h>
#include <cute/tensor.h>
#include <cute/tensor_generators.h>
#include <doctest/doctest.h>

namespace cute
{

__global__ void copy_increment(cute::TensorSpan<const int32_t, 2, Hardware::GPU> x,
                               cute::TensorSpan<int32_t, 2, Hardware::GPU> out,
                               int32_t increment)
{
    const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
    const auto idy = threadIdx.y + blockDim.y * blockIdx.y;
    if (idx < x.shape<0>(), idy < x.shape<1>())  // idx is valid given x's size on the first dimension
    {
        out.elem_ref(idy, idx) = x.elem(idy, idx) + increment;
    }
}

TEST_SUITE("Streams")
{
    TEST_CASE("Simple Increment Kernel works")
    {
        auto shape = cute::shape(2, 2);
        auto iterations = 10;
        const auto input = cute::iota<int32_t>(shape);
        const auto expected_out = cute::iota<int32_t>(shape, ((iterations) * (iterations - 1)) / 2);

        auto work = [&iterations, &input](StreamView<Hardware::GPU>& stream)
        {
            auto input_gpu = input.transfer_async<Hardware::GPU>(stream);
            for (auto i = 0; i < iterations; i++)
            {
                copy_increment<<<1, as_dim3(input.get_shape()), 0, stream>>>(input_gpu, input_gpu, i);
            }
            return input_gpu.template transfer_async<Hardware::CPU>(stream);
        };
        SUBCASE("Stream")
        {
            auto stream = cute::make_stream<Hardware::GPU>();
            auto res = work(stream);
            stream.synchronize();
            cute::equal(res, expected_out);
        }

        SUBCASE("Graph")
        {
            auto stream = cute::make_stream<Hardware::GPU>();
            auto graph = cute::Graph<Hardware::GPU>();

            auto recorder = graph.start_recording(stream);
            auto res = work(stream);
            recorder.stop_recording();

            auto instance = graph.get_instance();
            instance.launch(stream);
            stream.synchronize();
            cute::equal(res, expected_out);
        }
    }
}

}  // namespace cute
