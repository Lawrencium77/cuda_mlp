// CUDA kernels for Matrix class
#include "matrix_kernels.h"

__global__ void matrix_const_add(const float *a, const float value, float *output, const int rows, const int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        output[index] = a[index] + value;
    }
}

__global__ void matrix_const_mul(const float *a, const float value, float *output, const int rows, const int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        output[index] = a[index] * value;
    }
}

__global__ void matrix_sum(const float* data, float* sum, const int rows, const int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        float value = data[index];
        atomicAdd(sum, value);
    }
}

__global__ void matrix_max_abs(const float* data, float* max_abs, const int rows, const int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        float value = fabsf(data[index]);
        atomicMax(reinterpret_cast<int*>(max_abs), __float_as_int(value));
    }
}

__global__ void matrix_add(const float *a, const float *b, float *c, const int rows, const int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        c[index] = a[index] + b[index];
    }
}

__global__ void matrix_hadamard(const float *a, const float *b, float *c, const int rows, const int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        c[index] = a[index] * b[index];
    }
}

__global__ void matrix_transpose(const float *a, float *b, const int rows, const int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        b[col * rows + row] = a[row * cols + col];
    }
}

// cols_a = rows_b
__global__ void matrix_multiply(const float *a, const float *b, float *c, const int rows_a, const int cols_a, const int cols_b) {
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

__global__ void matrix_softmax_over_rows(const float *a, float* b, const int rows, const int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;

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

__global__ void matrix_sigmoid(const float *a, float* b, const int rows, const int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        b[index] = 1 / (1 + expf(-1 * a[index]));
    }
}

__global__ void matrix_relu(const float *a, float* b, const int rows, const int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        b[index] = fmaxf(0.0f, a[index]);
    }
}

__global__ void matrix_relu_backward(const float *a, const float *grad_output, float *grad_input, const int rows, const int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        grad_input[index] = a[index] > 0 ? grad_output[index] : 0.0f;
    }
}

// Random numbers drawn from uniform distribution
__global__ void fill_with_random(float *a, const unsigned long seed, const int rows, const int cols, const float min, const float max) {
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

__global__ void ce_loss(const float *preds, const float *labels, float *losses, const int rows, const int cols, const float epsilon) {
    int row = threadIdx.y;

    if (row < rows) {
        int label = (int)labels[row];
        float pred = preds[row * cols + label];
        losses[row] = -1 * logf(pred + epsilon);
    }
}

__global__ void ce_loss_and_predictions(const float *preds, const float *labels, float *losses, float *correct_predictions, const int rows, const int cols, const float epsilon) {
    int row = threadIdx.y;

    if (row < rows) {
        int label = (int)labels[row];
        float pred = preds[row * cols + label];
        losses[row] = -1 * logf(pred + epsilon);

        int predicted_label = 0;
        float max_prob = preds[row * cols];
        for (int col = 1; col < cols; ++col) {
            float current_prob = preds[row * cols + col];
            if (current_prob > max_prob) {
                max_prob = current_prob;
                predicted_label = col;
            }
        }
        correct_predictions[row] = (predicted_label == label) ? 1.0f : 0.0f;
    }
}

__global__ void softmax_bwd(const float* labels, const float* softmax_outputs, float* softmax_grads, const int rows, const int cols) {
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
