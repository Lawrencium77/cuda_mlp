#ifndef MATRIX_KERNELS_H
#define MATRIX_KERNELS_H

#ifdef __CUDACC__
#include <curand_kernel.h>
__global__ void matrix_const_add(float *a, float value, float *output, int rows, int cols);
__global__ void matrix_const_mul(float *a, float value, float *output, int rows, int cols);
__global__ void matrix_sum(float* data, float* sum, int rows, int cols);
__global__ void matrix_add(float *a, float *b, float *c, int rows, int cols);
__global__ void matrix_hadamard(float *a, float *b, float *c, int rows, int cols);
__global__ void matrix_transpose(float *a, float *b, int rows, int cols);
__global__ void matrix_multiply(float *a, float *b, float *c, int rows_a, int cols_a, int cols_b);
__global__ void matrix_softmax(float *a, float* b, int rows, int cols);
__global__ void matrix_sigmoid(float *a, float* b, int rows, int cols);
__global__ void fill_with_random(float *a, unsigned long seed, int rows, int cols);
__global__ void ce_loss(float *preds, float *labels, float *losses, int rows, int cols);
__global__ void softmax_bwd(float* label, float* softmax_output, float* softmax_grads, int rows, int cols);
#endif
#endif
