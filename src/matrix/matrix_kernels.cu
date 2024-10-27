// CUDA kernels for Matrix class
#include "matrix_kernels.h"

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
__global__ void matrix_softmax(float *a, float* b, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < cols) {
        float col_sum = 0.0f;
        for (int row = 0; row < rows; row++) {
            float exp_value = expf(a[col * rows + row]);
            b[col * rows + row] = exp_value;
            col_sum += exp_value;
        }

        for (int row = 0; row < rows; row++) {
            b[col * rows + row] /= col_sum;
        }
    }
}

__global__ void matrix_sigmoid(float *a, float* b, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        b[index] = 1 / (1 + expf(-1 * a[index]));
    }
}

// Random numbers drawn from normal distribution
__global__ void fill_with_random(float *a, unsigned long seed, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        curandState_t state;
        curand_init(seed, index, 0, &state);

        float rand_normal = curand_normal(&state);
        a[index] = rand_normal;
    }
}

__global__ void ce_loss(float *preds, float *labels, float *losses, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Row = 0
    
    if (col < cols) {
        int label = (int)labels[col];
        losses[col] = -1 * logf(preds[col * rows + label]);
    }
}
