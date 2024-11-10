#ifdef __CUDACC__
#include <cuda_runtime.h>

class MemoryAllocator {
    struct Block {
        void* ptr;
        size_t size;
        bool free;
        Block* next;
        Block* prev;

        Block(void* p, size_t s);
    };

    Block* head;
    size_t align_size(size_t size);

public:
    MemoryAllocator();
    ~MemoryAllocator();

    void* allocate(size_t requested_size);
    void free(void* ptr);

    void allocate_upfront(size_t requested_size);
};

#endif
