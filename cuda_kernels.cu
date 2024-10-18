// vector_add.cu
#include <iostream>

__global__ void vector_add(int *a, int *b, int *c, int n) {
    int index = threadIdx.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

__global__ void matrix_add(int *a, int *b, int *c, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int index = row * width + col;

    if (row < height && col < width) {
        c[index] = a[index] + b[index];
    }
}


int main() {
    const int N = 5;
    int a[N], b[N], c[N];
    int *d_a, *d_b, *d_c;

    // Fill arrays a and b with some values
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_b, N * sizeof(int));
    cudaMalloc((void**)&d_c, N * sizeof(int));

    // Copy data from host to device (CPU to GPU)
    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the add() kernel on the GPU
    add<<<1, N>>>(d_a, d_b, d_c, N);

    // Copy the result from device to host (GPU to CPU)
    cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "Result: ";
    for (int i = 0; i < N; i++) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
