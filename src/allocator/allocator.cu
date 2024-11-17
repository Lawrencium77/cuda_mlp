// Simple device memory allocator.
// Uses a doubly-linked list, and first-fit strategy for finding free blocks.
// Also implements free block coalescing.
// Inspired by https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html
#include "allocator.h"
#include <iostream>

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

// TODO: Implement block splitting to improve memory utilisation
void* MemoryAllocator::allocate(size_t requested_size) {
    if (requested_size == 0) {
        return nullptr;
    }
    size_t size = align_size(requested_size);

    Block* current = head;
    while (current != nullptr) {
        if (current->free && current->size >= size) {
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

    Block* current = head;
    while (current != nullptr) {
        if (current->ptr == ptr) {
            current->free = true;

            // Coalesce with next block(s)
            while (
                current->next &&
                current->next->free &&
                (static_cast<char*>(current->ptr) + current->size == current->next->ptr) // Blocks aren't guaranteed to be contiguous in memory, so we must check
            ) {
                Block* next_block = current->next;
                current->size += next_block->size;
                current->next = next_block->next;
                if (next_block->next) {
                    next_block->next->prev = current;
                }
                delete next_block;
            }


            // Coalesce with previous block(s)
            while (
                current->prev &&
                current->prev->free &&
                (static_cast<char*>(current->prev->ptr) + current->prev->size == current->ptr)
            ) {
                Block* prev_block = current->prev;
                prev_block->size += current->size;
                prev_block->next = current->next;
                if (current->next) {
                    current->next->prev = prev_block;
                }
                delete current;
                current = prev_block;
            }

            return;
        }
        current = current->next;
    }

    // If we didn't find the block, raise an error
    std::cerr << "Attempted to call free(ptr) but ptr does not correspond to a memory block" << std::endl;
    exit(1);
}