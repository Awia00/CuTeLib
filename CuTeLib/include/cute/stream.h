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
    // no specific implementation for CPU
};

#ifdef __CUDACC__

// static int32_t STREAM_GPU_LEAST_PRIORITY;
// static int32_t STREAM_GPU_GREATEST_PRIORITY;
// static auto foo = cudaDeviceGetStreamPriorityRange(&STREAM_GPU_LEAST_PRIORITY, &STREAM_GPU_GREATEST_PRIORITY);
// static int32_t STREAM_GPU_DEFAULT_PRIORITY = (STREAM_GPU_LEAST_PRIORITY + STREAM_GPU_GREATEST_PRIORITY) / 2;

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
    Stream(uint32_t flags, int32_t priority)
    {
        cudaStreamCreateWithPriority(&this->native_stream_, flags, priority);
    }

    Stream(Stream&) = delete;
    Stream<Hardware::GPU>& operator=(Stream&) = delete;

    Stream(Stream&& o_stream) : native_stream_(std::move(o_stream.native_stream_))
    {
        o_stream.native_stream_ = nullptr;
    }
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

    void wait_for(Event<Hardware::GPU>& event, uint32_t flags = 0);

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

    Event(uint32_t flags)
    {
        cudaEventCreateWithFlags(&this->native_event_, flags);
    }
    Event(Event&) = delete;
    Event<Hardware::GPU>& operator=(Event&) = delete;

    Event(Event&& o_event) : native_event_(std::move(o_event.native_event_))
    {
        o_event.native_event_ = nullptr;
    }
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

void Stream<Hardware::GPU>::wait_for(Event<Hardware::GPU>& event, uint32_t flags)
{
    cudaStreamWaitEvent(this->native_stream_, event, flags);
}

#endif


} // namespace cute
