#ifndef KERNELS_H
#define KERNELS_H

#ifdef __CUDACC__
__global__ void vector_add_const(float *a, float value, float *c, int n);
#endif

#endif
