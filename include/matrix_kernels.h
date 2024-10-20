#ifndef MATRIX_KERNELS_H
#define MATRIX_KERNELS_H

#ifdef __CUDACC__
__global__ void matrix_add(float *a, float *b, float *c, int rows, int cols);
__global__ void matrix_multiply(float *a, float *b, float *c, int rows_a, int cols_a, int cols_b);
__global__ void matrix_softmax(float *a, float* b, int rows_a, int cols_a);
#endif
#endif
