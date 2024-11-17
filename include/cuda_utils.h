#ifdef __CUDACC__
#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <cstdlib>

void checkCudaState(const std::string& file, int line);
void checkCudaState(cudaError_t err, const std::string& file, int line);

#define CHECK_CUDA_STATE() checkCudaState(__FILE__, __LINE__)
#define CHECK_CUDA_STATE_WITH_ERR(err) checkCudaState((err), __FILE__, __LINE__)

#endif // CUDA_UTILS_H
#endif // __CUDACC__
