#pragma once

#include <iostream>
#include <utility>
#include <cute/errors.h>
#include <cute/stream.h>

namespace cute
{


#ifdef __CUDACC__

struct Graph;

struct GraphInstance
{
    private:
    cudaGraphExec_t native_instance_;

    public:
    explicit GraphInstance(const Graph& graph);

    GraphInstance(const GraphInstance&) = delete;
    GraphInstance& operator=(const GraphInstance&) = delete;
    GraphInstance(GraphInstance&& other) : native_instance_(std::move(other.native_instance_))
    {
        other.native_instance_ = nullptr;
    }
    GraphInstance& operator=(GraphInstance&& other)
    {
        cudaGraphExecDestroy(this->native_instance_);
        this->native_instance_ = std::move(other.native_instance_);
        other.native_instance_ = nullptr;
        return *this;
    }
    ~GraphInstance()
    {
        CUTE_ERROR_NOTIFY(cudaGraphExecDestroy(this->native_instance_));
    }

    void launch(StreamView& stream)
    {
        CUTE_ERROR_CHECK(cudaGraphLaunch(this->native_instance_, stream));
    }
};


struct Graph
{
    private:
    cudaGraph_t native_graph_;

    public:
    Graph() noexcept(CUTE_NOEXCEPT) : native_graph_()
    {
        constexpr auto flag = 0;  // flag must be 0, other values are only used for CUDA graphs
        CUTE_ERROR_CHECK(cudaGraphCreate(&this->native_graph_, flag));
    }

    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
    Graph(Graph&& other) noexcept(true) : native_graph_(std::move(other.native_graph_))
    {
        other.native_graph_ = nullptr;
    }
    Graph& operator=(Graph&& other) noexcept(CUTE_NOEXCEPT)
    {
        CUTE_ERROR_CHECK(cudaGraphDestroy(this->native_graph_));
        this->native_graph_ = std::move(other.native_graph_);
        other.native_graph_ = nullptr;
        return *this;
    }

    ~Graph()
    {
        CUTE_ERROR_NOTIFY(cudaGraphDestroy(this->native_graph_));
    }

    [[nodiscard]] GraphInstance get_instance() && = delete;
    [[nodiscard]] GraphInstance get_instance() &
    {
        return GraphInstance(*this);
    }

    operator cudaGraph_t&()
    {
        return this->native_graph_;
    }

    operator const cudaGraph_t&() const
    {
        return this->native_graph_;
    }

    cudaGraph_t& get_native()
    {
        return this->native_graph_;
    }

    const cudaGraph_t& get_native() const
    {
        return this->native_graph_;
    }

    private:
    struct [[nodiscard]] StreamRecording
    {
        private:
        bool recording_{ true };
        StreamView& stream_;
        Graph& graph_;

        public:
        StreamRecording(StreamView& stream,
                        Graph& graph,
                        cudaStreamCaptureMode mode = cudaStreamCaptureModeGlobal) noexcept(CUTE_NOEXCEPT)
          : recording_(true), stream_(stream), graph_(graph)
        {
            CUTE_ERROR_CHECK(cudaStreamBeginCapture(this->stream_, mode));
        }

        StreamRecording(const StreamRecording&) = delete;
        StreamRecording(StreamRecording&&) = delete;
        StreamRecording& operator=(const StreamRecording&) = delete;
        StreamRecording& operator=(StreamRecording&&) = delete;
        ~StreamRecording()
        {
            try
            {
                this->stop_recording();
            }
            catch (const std::exception& ex)
            {
                std::cerr << "error occured in destructor: " << ex.what() << std::endl;
            }
        }

        void stop_recording() noexcept(CUTE_NOEXCEPT)
        {
            if (this->recording_)
            {
                CUTE_ERROR_CHECK(cudaStreamEndCapture(this->stream_, &this->graph_.get_native()));
                this->recording_ = false;
            }
        }
    };

    public:
    [[nodiscard]] StreamRecording start_recording(StreamView& stream, cudaStreamCaptureMode mode) && = delete;
    [[nodiscard]] StreamRecording start_recording(StreamView& stream,
                                                  cudaStreamCaptureMode mode = cudaStreamCaptureModeGlobal) &
    {
        return StreamRecording(stream, *this, mode);
    }
};


GraphInstance::GraphInstance(const Graph& graph) : native_instance_()
{
    CUTE_ERROR_CHECK(cudaGraphInstantiate(&this->native_instance_, graph, NULL, NULL, 0));
}

#endif
}  // namespace cute
