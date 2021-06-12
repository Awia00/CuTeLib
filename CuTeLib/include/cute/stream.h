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
class EventView;
class StreamView;

#ifdef __CUDACC__

class StreamView
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
    void wait_for(EventView& event);

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
    static int32_t get_lowest_priority()
    {
        auto priority = 0;
        cudaDeviceGetStreamPriorityRange(&priority, nullptr);
        return priority;
    }
    static int32_t get_greatest_priority()
    {
        auto priority = 0;
        cudaDeviceGetStreamPriorityRange(nullptr, &priority);
        return priority;
    }
};

/**
 * @brief Stream is an owning version of Stream.
 *
 */
class Stream : public StreamView
{
    using BaseT = StreamView;

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

#endif

// Events

#ifdef __CUDACC__
constexpr uint32_t EVENT_GPU_FLAGS_MAX_PERFORMANCE = cudaEventDisableTiming;

class EventView
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
    void record(StreamView& stream)
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

inline void StreamView::wait_for(EventView& event)
{
    constexpr auto flag = 0;  // flag must be 0, other values are only used for CUDA graphs
    cudaStreamWaitEvent(this->native_stream_, event, flag);
}

class Event : public EventView
{
    using BaseT = EventView;

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
     * @return Event&
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
