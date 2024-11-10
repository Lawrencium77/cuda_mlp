// Simple device memory allocator.
// Uses a doubly-linked list, and first-fit strategy for finding free blocks.
// Also implements free block coalescing.
// Inspired by https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html
#include "allocator.h"

MemoryAllocator::Block::Block(void* p, size_t s) : ptr(p), size(s), free(true), next(nullptr), prev(nullptr) {}
MemoryAllocator::MemoryAllocator() : head(nullptr) {}
MemoryAllocator::~MemoryAllocator() {
    Block* current = head;
    while (current != nullptr) {
        Block* next = current->next;
        cudaFree(current->ptr);
        delete current;
        current = next;
    }
}

// Round up to nearest multiple of 256 bytes for alignment
size_t MemoryAllocator::align_size(size_t size) {
    return (size + 255) & ~255;
}

void* MemoryAllocator::allocate(size_t requested_size) {
    if (requested_size == 0) {
        return nullptr;
    }
    size_t size = align_size(requested_size);

    Block* current = head;
    while (current != nullptr) {
        if (current->free && current->size >= size) {
            // Split block if it's way bigger than requested_size
            if (current->size > size + 256) {
                void* new_ptr = static_cast<char*>(current->ptr) + size;
                Block* new_block = new Block(new_ptr, current->size - size);

                new_block->next = current->next;
                new_block->prev = current;
                if (current->next) {
                    current->next->prev = new_block;
                }
                current->next = new_block;
                current->size = size;
            }

            current->free = false;
            return current->ptr;
        }
        current = current->next;
    }

    // No suitable block found
    void* new_ptr;
    cudaMalloc(&new_ptr, size);

    Block* new_block = new Block(new_ptr, size);
    new_block->free = false;
    new_block->next = head;
    if (head) {
        head->prev = new_block;
    }
    head = new_block;

    return new_ptr;
}

void MemoryAllocator::free(void* ptr) {
    if (!ptr) return;

    // Find block containing this pointer
    Block* current = head;
    while (current != nullptr) {
        if (current->ptr == ptr) {
            current->free = true;

            // Maybe coalesce with next block
            if (current->next && current->next->free) {
                current->size += current->next->size;
                Block* to_delete = current->next;
                current->next = to_delete->next;
                if (to_delete->next) {
                    to_delete->next->prev = current;
                }
                delete to_delete;
            }

            // Maybe coalesce with prev block
            if (current->prev && current->prev->free) {
                current->prev->size += current->size;
                current->prev->next = current->next;
                if (current->next) {
                    current->next->prev = current->prev;
                }
                delete current;
            }

            return;
        }
        current = current->next;
    }

    // If we didn't find the block, just call cudaFree
    cudaFree(ptr);
}

// Allocate and immediately free requested_size bytes.
// Useful when doing one large allocation at the start of the program.
// This minimises time spent doign cudaMalloc.
void MemoryAllocator::allocate_upfront(size_t requested_size) {
    void* allocator_head = allocate(requested_size);
    free(allocator_head);
};
