#ifndef MATRIX_KERNELS_H
#define MATRIX_KERNELS_H

#ifdef __CUDACC__
#include <curand_kernel.h>
__global__ void matrix_const_add(const float *a, const float value, float *output, const int rows, const int cols);
__global__ void matrix_const_mul(const float *a, const float value, float *output, const int rows, const int cols);
__global__ void matrix_sum(const float* data, float* sum, const int rows, const int cols);
__global__ void matrix_max_abs(const float* data, float* sum, const int rows, const int cols);
__global__ void matrix_add(const float *a, const float *b, float *c, const int rows, const int cols);
__global__ void matrix_hadamard(const float *a, const float *b, float *c, const int rows, const int cols);
__global__ void matrix_transpose(const float *a, float *b, const int rows, const int cols);
__global__ void matrix_multiply(const float *a, const float *b, float *c, const int rows_a, const int cols_a, const int cols_b);
__global__ void matrix_softmax_over_rows(const float *a, float* b, const int rows, const int cols);
__global__ void matrix_sigmoid(const float *a, float* b, const int rows, const int cols);
__global__ void matrix_relu(const float *a, float* b, const int rows, const int cols);
__global__ void matrix_relu_backward(const float *a, const float *grad_output, float *grad_input, const int rows, const int cols);
__global__ void fill_with_random(float *a, const unsigned long seed, const int rows, const int cols, const float min, const float max);
__global__ void ce_loss(const float *preds, const float *labels, float *losses, const int rows, const int cols, const float epsilon = 1e-7);
__global__ void softmax_bwd(const float* labels, const float* softmax_outputs, float* softmax_grads, const int rows, const int cols);
#endif
#endif
