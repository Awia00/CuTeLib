#pragma once

#include <utility>
#include <cute/stream.h>

namespace cute
{

template <Hardware HardwareV>
struct GraphInstance
{
    // no specific implementation for CPU - feel free to specialize
};


template <Hardware HardwareV>
struct Graph
{
    // no specific implementation for CPU - feel free to specialize
};

#ifdef __CUDACC__

template <>
struct GraphInstance<Hardware::GPU>
{
    using MyT = GraphInstance<Hardware::GPU>;

    private:
    cudaGraphExec_t native_instance_;

    public:
    explicit GraphInstance(const Graph<Hardware::GPU>& graph);

    GraphInstance(const MyT&) = delete;
    MyT& operator=(const MyT&) = delete;
    GraphInstance(MyT&& other) : native_instance_(std::move(other.native_instance_))
    {
        other.native_instance_ = nullptr;
    }
    MyT& operator=(MyT&& other)
    {
        cudaGraphExecDestroy(this->native_instance_);
        this->native_instance_ = std::move(other.native_instance_);
        other.native_instance_ = nullptr;
        return *this;
    }
    ~GraphInstance()
    {
        cudaGraphExecDestroy(this->native_instance_);
    }

    void launch(StreamView<Hardware::GPU>& stream)
    {
        cudaGraphLaunch(this->native_instance_, stream);
    }
};


template <>
struct Graph<Hardware::GPU>
{
    using MyT = Graph<Hardware::GPU>;

    private:
    cudaGraph_t native_graph_;

    public:
    Graph() : native_graph_()
    {
        constexpr auto flag = 0;  // flag must be 0, other values are only used for CUDA graphs
        cudaGraphCreate(&this->native_graph_, flag);
    }

    Graph(const MyT&) = delete;
    MyT& operator=(const MyT&) = delete;
    Graph(MyT&& other) : native_graph_(std::move(other.native_graph_))
    {
        other.native_graph_ = nullptr;
    }
    MyT& operator=(MyT&& other)
    {
        cudaGraphDestroy(this->native_graph_);
        this->native_graph_ = std::move(other.native_graph_);
        other.native_graph_ = nullptr;
        return *this;
    }

    ~Graph()
    {
        cudaGraphDestroy(this->native_graph_);
    }

    [[nodiscard]] GraphInstance<Hardware::GPU> get_instance()
    {
        return GraphInstance<Hardware::GPU>(*this);
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
    struct StreamRecording
    {
        private:
        bool recording_;
        StreamView<Hardware::GPU>& stream_;
        Graph<Hardware::GPU>& graph_;

        public:
        StreamRecording(StreamView<Hardware::GPU>& stream,
                        Graph<Hardware::GPU>& graph,
                        cudaStreamCaptureMode mode = cudaStreamCaptureModeGlobal)
          : recording_(true), stream_(stream), graph_(graph)
        {
            cudaStreamBeginCapture(this->stream_, mode);
        }

        StreamRecording(const StreamRecording&) = delete;
        StreamRecording(StreamRecording&&) = delete;
        StreamRecording& operator=(const StreamRecording&) = delete;
        StreamRecording& operator=(StreamRecording&&) = delete;
        ~StreamRecording()
        {
            this->stop_recording();
        }

        void stop_recording()
        {
            if (this->recording_)
            {
                cudaStreamEndCapture(this->stream_, &this->graph_.get_native());
                this->recording_ = false;
            }
        }
    };

    public:
    [[nodiscard]] StreamRecording start_recording(StreamView<Hardware::GPU>& stream,
                                                  cudaStreamCaptureMode mode = cudaStreamCaptureModeGlobal)
    {
        return StreamRecording(stream, *this, mode);
    }
};


GraphInstance<Hardware::GPU>::GraphInstance(const Graph<Hardware::GPU>& graph) : native_instance_()
{
    cudaGraphInstantiate(&this->native_instance_, graph, NULL, NULL, 0);
}

#endif
}  // namespace cute
