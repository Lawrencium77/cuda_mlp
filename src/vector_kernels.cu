// CUDA kernels for Vector class

__global__ void vector_add_const(float *a, float value, float *c, int n) {
    int index = threadIdx.x;
    if (index < n) {
        c[index] = a[index] + value;
    }
}

__global__ void vector_sub_const(float *a, float value, float *c, int n) {
    int index = threadIdx.x;
    if (index < n) {
        c[index] = a[index] - value;
    }
}

__global__ void vector_mul_const(float *a, float value, float *c, int n) {
    int index = threadIdx.x;
    if (index < n) {
        c[index] = a[index] * value;
    }
}

__global__ void vector_div_const(float *a, float value, float *c, int n) {
    int index = threadIdx.x;
    if (index < n) {
        c[index] = a[index] / value;
    }
}

__global__ void vector_add(float *a, float *b, float *c, int n){
    int index = threadIdx.x;
    if (index < n){
        c[index] = a[index] + b[index];
    }
}

__global__ void vector_sub(float *a, float *b, float *c, int n){
    int index = threadIdx.x;
    if (index < n){
        c[index] = a[index] - b[index];
    }
}

__global__ void vector_mul(float *a, float *b, float *c, int n){
    int index = threadIdx.x;
    if (index < n){
        c[index] = a[index] * b[index];
    }
}

__global__ void vector_div(float *a, float *b, float *c, int n){
    int index = threadIdx.x;
    if (index < n){
        c[index] = a[index] / b[index];
    }
}
