// CUDA kernels for Matrix class
#include "matrix_kernels.h"

__global__ void matrix_const_add(float *a, float value, float *output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        output[index] = a[index] + value;
    }
}

__global__ void matrix_const_mul(float *a, float value, float *output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        output[index] = a[index] * value;
    }
}

__global__ void matrix_sum(float* data, float* sum, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        float value = data[index];
        atomicAdd(sum, value);
    }
}


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

__global__ void matrix_transpose(float *a, float *b, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        b[col * rows + row] = a[row * cols + col];
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

__global__ void matrix_softmax_over_rows(float *a, float* b, int rows, int cols) {
    int row = threadIdx.y;

    if (row < rows) {
        float row_max = a[row * cols];
        for (int col = 1; col < cols; col++) {
            float val = a[row * cols + col];
            if (val > row_max) {
                row_max = val;
            }
        }

        float row_sum = 0.0f;
        for (int col = 0; col < cols; col++) {
            float exp_value = expf(a[row * cols + col] - row_max); // subtract max for stability
            b[row * cols + col] = exp_value;
            row_sum += exp_value;
        }

        for (int col = 0; col < cols; col++) {
            b[row * cols + col] /= row_sum;
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

__global__ void matrix_relu(float *a, float* b, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        b[index] = fmaxf(0.0f, a[index]);
    }
}

__global__ void matrix_relu_backward(float *a, float *grad_output, float *grad_input, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        grad_input[index] = a[index] > 0 ? grad_output[index] : 0.0f;
    }
}

// Random numbers drawn from uniform distribution
__global__ void fill_with_random(float *a, unsigned long seed, int rows, int cols, float min, float max) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        curandState_t state;
        curand_init(seed, index, 0, &state);

        float rand_uniform = curand_uniform(&state);
        a[index] = min + rand_uniform * (max - min); // Scale to range
    }
}

__global__ void ce_loss(float *preds, float *labels, float *losses, int rows, int cols, const float epsilon) {
    int row = threadIdx.y;

    if (row < rows) {
        int label = (int)labels[row];
        float pred = preds[row * cols + label];
        losses[row] = -1 * logf(pred + epsilon);
    }
}

__global__ void softmax_bwd(float* labels, float* softmax_outputs, float* softmax_grads, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        int label_idx = (int)labels[row];

        // https://shivammehta25.github.io/posts/deriving-categorical-cross-entropy-and-softmax/#derivation-of-softmax
        if (col == label_idx) {
            softmax_grads[idx] = softmax_outputs[idx] - 1.0f;
        } else {
            softmax_grads[idx] = softmax_outputs[idx];
        }
    }
}
