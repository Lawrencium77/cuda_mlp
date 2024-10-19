#ifndef KERNELS_H
#define KERNELS_H

#ifdef __CUDACC__
__global__ void vector_add_const(float *a, float value, float *c, int n);
__global__ void vector_sub_const(float *a, float value, float *c, int n);
__global__ void vector_mul_const(float *a, float value, float *c, int n);
__global__ void vector_div_const(float *a, float value, float *c, int n);
__global__ void vector_add(float *a, float *b, float *c, int n);
__global__ void vector_sub(float *a, float *b, float *c, int n);
__global__ void vector_mul(float *a, float *b, float *c, int n);
__global__ void vector_div(float *a, float *b, float *c, int n);
__global__ void vector_sigmoid(float *a, float*c, int n);
#endif
#endif
