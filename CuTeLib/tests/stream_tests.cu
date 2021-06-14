
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

        // cuda graphs still dont do well with allocations: so we have to receive them allocated
        auto work = [&iterations, &input](Tensor<int32_t, 2, Hardware::GPU>& in_gpu,
                                          Tensor<int32_t, 2, Hardware::CPU>& out,
                                          StreamView& stream)
        {
            copy_async(input, in_gpu, stream);
            for (auto i = 0; i < iterations; i++)
            {
                copy_increment<<<1, as_dim3(input.get_shape()), 0, stream>>>(in_gpu, in_gpu, i);
            }
            copy_async(in_gpu, out, stream);
        };

        SUBCASE("Stream")
        {
            auto input_gpu = Tensor<int32_t, 2, Hardware::GPU>(shape);
            auto out = Tensor<int32_t, 2, Hardware::CPU>(shape);
            auto stream = cute::Stream();
            work(input_gpu, out, stream);
            stream.synchronize();
            cute::equal(out, expected_out);
        }

        SUBCASE("Graph")
        {
            auto input_gpu = Tensor<int32_t, 2, Hardware::GPU>(shape);
            auto out = Tensor<int32_t, 2, Hardware::CPU>(shape);

            auto stream = cute::Stream();
            auto graph = cute::Graph();

            auto recorder = graph.start_recording(stream);
            work(input_gpu, out, stream);
            recorder.stop_recording();

            auto instance = graph.get_instance();
            instance.launch(stream);
            stream.synchronize();
            cute::equal(out, expected_out);
        }
    }
}

}  // namespace cute
