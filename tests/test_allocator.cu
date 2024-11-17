#include <iostream>
#include "allocator.h"

void test_basic_allocation_and_deallocation(MemoryAllocator& allocator) {
    std::cout << "Test Basic allocation and deallocation" << std::endl;
    void* ptr1 = allocator.allocate(1024); // 1KB
    if (!ptr1) {
        std::cerr << "Failed to allocate 1KB" << std::endl;
        exit(1);
    }
    allocator.free(ptr1);
}

void test_multiple_allocations_and_deallocations(MemoryAllocator& allocator) {
    std::cout << "Test Multiple allocations and deallocations" << std::endl;
    void* ptr2 = allocator.allocate(2048);
    void* ptr3 = allocator.allocate(4096);
    void* ptr4 = allocator.allocate(8192);
    if (!ptr2 || !ptr3 || !ptr4) {
        std::cerr << "Failed to allocate multiple blocks" << std::endl;
        exit(1);
    }
    allocator.free(ptr2);
    allocator.free(ptr3);
    allocator.free(ptr4);
}

void test_interleaved_allocation_and_deallocation(MemoryAllocator& allocator) {
    std::cout << "Test Interleaved allocation and deallocation" << std::endl;
    void* ptrA = allocator.allocate(1024);
    void* ptrB = allocator.allocate(2048);
    void* ptrC = allocator.allocate(4096);
    allocator.free(ptrB);
    void* ptrD = allocator.allocate(2048);
    if (ptrD != ptrB) {
        std::cerr << "Allocator did not reuse freed block" << std::endl;
        exit(1);
    }
    allocator.free(ptrA);
    allocator.free(ptrC);
    allocator.free(ptrD);
}

void test_block_coalescing(MemoryAllocator& allocator) {
    std::cout << "Test Block coalescing" << std::endl;
    void* ptrE = allocator.allocate(1024);
    void* ptrF = allocator.allocate(2048);
    void* ptrG = allocator.allocate(4096);
    allocator.free(ptrF);
    allocator.free(ptrG);
    void* ptrH = allocator.allocate(6144); // Should fit into coalesced block
    if (!ptrH) {
        std::cerr << "Failed to allocate in coalesced block" << std::endl;
        exit(1);
    }
    allocator.free(ptrE);
    allocator.free(ptrH);
}

void test_block_splitting(MemoryAllocator& allocator) {
    std::cout << "Test Block splitting" << std::endl;
    void* ptrI = allocator.allocate(8192);
    allocator.free(ptrI);
    void* ptrJ = allocator.allocate(1024);
    if (!ptrJ) {
        std::cerr << "Failed to split block" << std::endl;
        exit(1);
    }
    allocator.free(ptrJ);
}

void test_zero_byte_allocation(MemoryAllocator& allocator) {
    std::cout << "Test Zero-byte allocation" << std::endl;
    void* ptrZero = allocator.allocate(0);
    if (ptrZero != nullptr) {
        std::cerr << "Allocator should return nullptr for zero-byte allocation" << std::endl;
        exit(1);
    }
}

void test_free_nullptr(MemoryAllocator& allocator) {
    std::cout << "Test Freeing nullptr" << std::endl;
    allocator.free(nullptr);
}

void test_stress_random_sizes(MemoryAllocator& allocator) {
    std::cout << "Test Stress test with random sizes" << std::endl;
    const int num_allocs = 1000;
    void* allocations[num_allocs];
    for (int i = 0; i < num_allocs; ++i) {
        size_t size = (rand() % 1024) + 1;
        allocations[i] = allocator.allocate(size);
        if (!allocations[i]) {
            std::cerr << "Failed to allocate in stress test at iteration " << i << std::endl;
            exit(1);
        }
    }
    for (int i = 0; i < num_allocs; ++i) {
        allocator.free(allocations[i]);
    }
}

int main() {
    MemoryAllocator allocator;

    test_basic_allocation_and_deallocation(allocator);
    test_multiple_allocations_and_deallocations(allocator);
    test_interleaved_allocation_and_deallocation(allocator);
    test_block_coalescing(allocator);
    test_block_splitting(allocator);
    test_zero_byte_allocation(allocator);
    test_free_nullptr(allocator);
    test_stress_random_sizes(allocator);

    allocator.cleanup();

    return 0;
}
