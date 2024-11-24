#ifndef MATRIX_KERNELS_H
#define MATRIX_KERNELS_H

#ifdef __CUDACC__
#include <algorithm>
#include <curand_kernel.h>
template <typename T>
__global__ void matrix_const_add(const T* a, const T value, T* output, const int rows, const int cols);

template <typename T>
__global__ void matrix_const_mul(const T* a, const T value, T* output, const int rows, const int cols);

template <typename T>
__global__ void matrix_sum(const T* data, float* sum, const int rows, const int cols);

template <typename T>
__global__ void matrix_add(const T* a, const T* b, T* c, const int rows, const int cols);

template <typename T>
__global__ void matrix_hadamard(const T* a, const T* b, T* c, const int rows, const int cols);

template <typename T>
__global__ void matrix_transpose(const T* a, T* b, const int rows, const int cols);

template <typename T>
__global__ void matrix_multiply(const T* a, const T* b, T* c, const int rows_a, const int cols_a, const int cols_b);

template <typename T>
__global__ void matrix_softmax_over_rows(const T* a, T* b, const int rows, const int cols);

template <typename T>
__global__ void matrix_sigmoid(const T* a, T* b, const int rows, const int cols);

template <typename T>
__global__ void matrix_relu(const T* a, T* b, const int rows, const int cols);

template <typename T>
__global__ void matrix_relu_backward(const float* a, const float* grad_output, float* grad_input, const int rows, const int cols);

template <typename T>
__global__ void fill_with_random(T* a, const unsigned long seed, const int rows, const int cols, const T min, const T max);

// TODO: Sort out labels dtype so it's always int
template <typename T>
__global__ void ce_loss(const T* preds, const T* labels, T* losses, const int rows, const int cols, const float epsilon = 1e-7);

template <typename T>
__global__ void ce_loss_and_predictions(const T* preds, const T* labels, T* losses, T* correct_predictions, const int rows, const int cols, const float epsilon = 1e-7);

template <typename T>
__global__ void softmax_bwd(const T* labels, const T* softmax_outputs, T* softmax_grads, const int rows, const int cols);

__global__ void convertFP32ToFP16(half* out, float* in, int numel);
__global__ void convertFP16ToFP32(float* out, half* in, int numel);

#include "matrix_kernels.cuh"

#endif
#endif
