#pragma once

#include <memory>
#include <utility>
#include <cute/defs.h>
#include <cute/hardware.h>
#ifdef __CUDACC__
#include <cuda.h>
#endif


// Streams
namespace cute
{

template <Hardware HardwareV>
class EventView;

template <Hardware HardwareV>
class StreamView
{
    // no specific implementation for CPU - feel free to specialize
    public:
    void synchronize()
    {
    }
};

template <Hardware HardwareV>
class Stream : public StreamView<HardwareV>
{
};

#ifdef __CUDACC__

template <>
class StreamView<Hardware::GPU>
{
    protected:
    cudaStream_t native_stream_;

    StreamView() : native_stream_()
    {
    }

    public:
    explicit StreamView(cudaStream_t stream) : native_stream_(stream)
    {
    }

    StreamView(StreamView&) = delete;
    StreamView& operator=(StreamView&) = delete;
    StreamView(StreamView&& other) : native_stream_(std::move(other.native_stream_))
    {
        other.native_stream_ = nullptr;
    }
    StreamView& operator=(StreamView&& other)
    {
        this->native_stream_ = other.native_stream_;
        other.native_stream_ = nullptr;
        return *this;
    }

    protected:
    public:
    void synchronize()
    {
        cudaStreamSynchronize(this->native_stream_);
    }

    /**
     * @brief on the stream, enqueue waiting for the event given as argument.
     *
     * @param event
     */
    void wait_for(EventView<Hardware::GPU>& event);

    operator cudaStream_t&()
    {
        return this->native_stream_;
    }

    operator const cudaStream_t&() const
    {
        return this->native_stream_;
    }

    /**
     * @brief Returns the dedicated stream-per-thread
     *
     * @return StreamView<HardwareV>
     */
    static StreamView stream_per_thread()
    {
        return StreamView(cudaStreamPerThread);
    }
};

/**
 * @brief Stream is an owning version of Stream.
 *
 */
template <>
class Stream<Hardware::GPU> : public StreamView<Hardware::GPU>
{
    using BaseT = StreamView<Hardware::GPU>;

    public:
    Stream() : BaseT()
    {
        cudaStreamCreate(&this->native_stream_);
    }

    explicit Stream(uint32_t flags)
    {
        cudaStreamCreateWithFlags(&this->native_stream_, flags);
    }
    /**
     * @brief Construct a new Stream object. For priority see 'cudaDeviceGetStreamPriorityRange(int& priority_low, int& priority_high)'
     *
     * @param flags
     * @param priority
     */
    Stream(uint32_t flags, int32_t priority)
    {
        cudaStreamCreateWithPriority(&this->native_stream_, flags, priority);
    }


    Stream(Stream&) = delete;
    Stream& operator=(Stream&) = delete;

    Stream(Stream&& o_stream) noexcept : BaseT(std::move(o_stream.native_stream_))
    {
        o_stream.native_stream_ = nullptr;
    }

    /**
     * @brief Stream move assignment constructor. Notice that the current native stream will be destroyed.
     *
     * @param o_stream
     * @return Stream&
     */
    Stream& operator=(Stream&& o_stream) noexcept
    {
        cudaStreamDestroy(this->native_stream_);
        this->native_stream_ = std::move(o_stream.native_stream_);
        o_stream.native_stream_ = nullptr;
        return *this;
    }

    ~Stream()
    {
        cudaStreamDestroy(this->native_stream_);
    }
};

template <Hardware HardwareV>
Stream<HardwareV> make_stream()
{
    return Stream<HardwareV>();
}

template <Hardware HardwareV>
Stream<HardwareV> make_stream(uint32_t flags)
{
    return Stream<HardwareV>(flags);
}

template <Hardware HardwareV>
Stream<HardwareV> make_stream(uint32_t flags, int32_t priority)
{
    return Stream<HardwareV>(flags, priority);
}

#endif

// Events

template <Hardware HardwareV>
class EventView
{
    // no specific implementation for CPU
};

template <Hardware HardwareV>
class Event : public EventView<HardwareV>
{
    // no specific implementation for CPU
};

#ifdef __CUDACC__
constexpr uint32_t EVENT_GPU_FLAGS_MAX_PERFORMANCE = cudaEventDisableTiming;

template <>
class EventView<Hardware::GPU>
{
    protected:
    cudaEvent_t native_event_;

    EventView() : native_event_()
    {
    }

    public:
    explicit EventView(cudaEvent_t event) : native_event_(event)
    {
    }

    /**
     * @brief Records an event (this) on the stream
     *
     * @param stream
     */
    void record(StreamView<Hardware::GPU>& stream)
    {
        cudaEventRecord(this->native_event_, stream);
    }

    void synchronize()
    {
        cudaEventSynchronize(this->native_event_);
    }

    operator cudaEvent_t&()
    {
        return this->native_event_;
    }

    operator const cudaEvent_t&() const
    {
        return this->native_event_;
    }
};

inline void StreamView<Hardware::GPU>::wait_for(EventView<Hardware::GPU>& event)
{
    constexpr auto flag = 0;  // flag must be 0, other values are only used for CUDA graphs
    cudaStreamWaitEvent(this->native_stream_, event, flag);
}

template <>
class Event<Hardware::GPU> : public EventView<Hardware::GPU>
{
    using BaseT = EventView<Hardware::GPU>;

    public:
    Event() : BaseT()
    {
        cudaEventCreate(&this->native_event_);
    }

    /**
     * @brief Construct a new Event object
     *
     * @param flags for max performance pass in cudaEventDisableTiming as per CUDA documentation
     */
    explicit Event(uint32_t flags)
    {
        cudaEventCreateWithFlags(&this->native_event_, flags);
    }
    Event(Event&) = delete;
    Event& operator=(Event&) = delete;

    Event(Event&& o_event) noexcept : BaseT(std::move(o_event.native_event_))
    {
        o_event.native_event_ = nullptr;
    }

    /**
     * @brief Move assignment constructor. Notice that the current native event will be destroyed.
     *
     * @param o_event
     * @return Event<Hardware::GPU>&
     */
    Event& operator=(Event&& o_event)
    {
        cudaEventDestroy(this->native_event_);
        this->native_event_ = std::move(o_event.native_event_);
        o_event.native_event_ = nullptr;
        return *this;
    }

    ~Event()
    {
        cudaEventDestroy(this->native_event_);
    }
};

#endif


}  // namespace cute
