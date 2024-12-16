#include "cuda_utils.h"
#include <cstddef>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

class AllocatorBase {
public:
    virtual ~AllocatorBase() = default;
    virtual void* allocate(size_t size) = 0;
    virtual void free(void* ptr) = 0;
    virtual void cleanup() {}

protected:
    AllocatorBase() = default;
};


class CudaAsyncAllocator : public AllocatorBase {
public:
    CudaAsyncAllocator();
    ~CudaAsyncAllocator();
    void* allocate(size_t size);
    void free(void* ptr);
};


class MemoryAllocator : public AllocatorBase {
private:
  struct Block {
    void *ptr;
    size_t size;
    bool free;
    Block *next;
    Block *prev;

    Block(void *p, size_t s);
  };

  Block *head;
  bool cleaned_up;

  size_t align_size(size_t size);

public:
  MemoryAllocator();
  ~MemoryAllocator();

  // Use cleanup function instead of destructor to free GPU memory
  // Destructor is called after CUDA driver shuts down, which is too late
  // cleanup() should be called before the driver shuts down but after all
  // Matrix objects are destroyed
  void cleanup();

  void *allocate(size_t requested_size);
  void free(void *ptr);
};
