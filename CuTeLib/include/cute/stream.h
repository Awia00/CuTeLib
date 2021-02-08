#pragma once
#include <cute/defs.h>
#include <cute/hardware.h>
#ifdef __CUDACC__
#include <cuda.h>
#endif


// Streams
namespace cute
{

template <Hardware HardwareV>
class Event;

template <Hardware HardwareV>
class Stream
{
    // no specific implementation for CPU - feel free to specialize
    public:
    void synchronize()
    {
    }
};

#ifdef __CUDACC__

template <>
class Stream<Hardware::GPU>
{
    cudaStream_t native_stream_;

    public:
    Stream()
    {
        cudaStreamCreate(&this->native_stream_);
    }
    Stream(uint32_t flags)
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
    Stream<Hardware::GPU>& operator=(Stream&) = delete;

    Stream(Stream&& o_stream) noexcept : native_stream_(std::move(o_stream.native_stream_))
    {
        o_stream.native_stream_ = nullptr;
    }
    /**
     * @brief Stream move assignment constructor. Notice that the current native stream will be destroyed.
     *
     * @param o_stream
     * @return Stream<Hardware::GPU>&
     */
    Stream<Hardware::GPU>& operator=(Stream&& o_stream)
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

    // interface

    void synchronize()
    {
        cudaStreamSynchronize(this->native_stream_);
    }

    /**
     * @brief on the stream, enqueue waiting for the event given as argument.
     *
     * @param event
     */
    void wait_for(Event<Hardware::GPU>& event);

    operator cudaStream_t&()
    {
        return this->native_stream_;
    }

    operator const cudaStream_t&() const
    {
        return this->native_stream_;
    }
};

#endif

// Events

template <Hardware HardwareV>
class Event
{
    // no specific implementation for CPU
};


#ifdef __CUDACC__
constexpr uint32_t EVENT_GPU_FLAGS_MAX_PERFORMANCE = cudaEventDisableTiming;

template <>
class Event<Hardware::GPU>
{
    cudaEvent_t native_event_;

    public:
    Event()
    {
        cudaEventCreate(&this->native_event_);
    }

    /**
     * @brief Construct a new Event object
     *
     * @param flags for max performance pass in cudaEventDisableTiming as per CUDA documentation
     */
    Event(uint32_t flags)
    {
        cudaEventCreateWithFlags(&this->native_event_, flags);
    }
    Event(Event&) = delete;
    Event<Hardware::GPU>& operator=(Event&) = delete;

    Event(Event&& o_event) noexcept : native_event_(std::move(o_event.native_event_))
    {
        o_event.native_event_ = nullptr;
    }

    /**
     * @brief Move assignment constructor. Notice that the current native event will be destroyed.
     *
     * @param o_event
     * @return Event<Hardware::GPU>&
     */
    Event<Hardware::GPU>& operator=(Event&& o_event)
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

    /**
     * @brief Records an event (this) on the stream
     *
     * @param stream
     */
    void record(Stream<Hardware::GPU>& stream)
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

void Stream<Hardware::GPU>::wait_for(Event<Hardware::GPU>& event)
{
    constexpr auto flag = 0; // flag must be 0, other values are only used for CUDA graphs
    cudaStreamWaitEvent(this->native_stream_, event, flag);
}

#endif


} // namespace cute
