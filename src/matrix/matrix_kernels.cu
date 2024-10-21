// CUDA kernels for Matrix class

__global__ void matrix_add(float *a, float *b, float *c, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        c[index] = a[index] + b[index];
    }
}

__global__ void matrix_hadamard(float *a, float *b, float *c, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        c[index] = a[index] * b[index];
    }
}

// cols_a = rows_b
__global__ void matrix_multiply(float *a, float *b, float *c, int rows_a, int cols_a, int cols_b) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows_a && col < cols_b) {
        float sum = 0.0f;
        for (int k = 0; k < cols_a; k++) {
            sum += a[row * cols_a + k] * b[k * cols_b + col];
        }
        c[row * cols_b + col] = sum;
    }
}

// rows_a = len(labels)
__global__ void matrix_softmax(float *a, float* b, int rows_a, int cols_a) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows_a) {
        float row_sum = 0.0f;
        for (int col = 0; col < cols_a; col++) {
            float exp_value = expf(a[row * cols_a + col]);
            b[row * cols_a + col] = exp_value;
            row_sum += exp_value;
        }

        for (int col = 0; col < cols_a; col++) {
            b[row * cols_a + col] /= row_sum;
        }
    }
}
