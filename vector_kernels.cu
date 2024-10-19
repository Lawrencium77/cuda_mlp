
__global__ void vector_add_const(float *a, float value, float *c, int n) {
    int index = threadIdx.x;
    if (index < n) {
        c[index] = a[index] + value;
    }
}
