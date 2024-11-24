template <typename T>
__global__ void matrix_const_add(const T* a, const T value, T* output, const int rows, const int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        output[index] = a[index] + value;
    }
}

template <typename T>
__global__ void matrix_const_mul(const T* a, const T value, T* output, const int rows, const int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        output[index] = a[index] * value;
    }
}

template <typename T>
__global__ void matrix_sum(const T* data, float* sum, const int rows, const int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        float value = static_cast<float>(data[index]);
        atomicAdd(sum, value);
    }
}

template <typename T>
__global__ void matrix_add(const T* a, const T* b, T* c, const int rows, const int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        c[index] = a[index] + b[index];
    }
}

template <typename T>
__global__ void matrix_hadamard(const T* a, const T* b, T* c, const int rows, const int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        c[index] = a[index] * b[index];
    }
}

template <typename T>
__global__ void matrix_transpose(const T *a, T *b, const int rows, const int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        b[col * rows + row] = a[row * cols + col];
    }
}

// cols_a = rows_b
template <typename T>
__global__ void matrix_multiply(const T* a, const T* b, T* c, const int rows_a, const int cols_a, const int cols_b) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows_a && col < cols_b) {
        float sum = 0.0f;
        if constexpr (std::is_same_v<T, __half>) {
            for (int k = 0; k < cols_a; k++) {
                sum += __half2float(a[row * cols_a + k]) * __half2float(b[k * cols_b + col]);
            }
            c[row * cols_b + col] = __float2half(sum);
        } else {
            for (int k = 0; k < cols_a; k++) {
                sum += a[row * cols_a + k] * b[k * cols_b + col];
            }
            c[row * cols_b + col] = sum;
        }
    }
}

template <typename T>
__global__ void matrix_softmax_over_rows(const T* a, T* b, const int rows, const int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows) {
        float row_max = static_cast<float>(a[row * cols]);
        for (int col = 1; col < cols; col++) {
            float val = static_cast<float>(a[row * cols + col]);
            if (val > row_max) {
                row_max = val;
            }
        }

        float row_sum = 0.0f;
        for (int col = 0; col < cols; col++) {
            float val = static_cast<float>(a[row * cols + col]);
            row_sum += expf(val - row_max); // subtract max for stability
        }

        for (int col = 0; col < cols; col++) {
            float val = static_cast<float>(a[row * cols + col]);
            float exp_value = expf(val - row_max); // subtract max for stability
            float softmax_value = exp_value / row_sum;
            b[row * cols + col] = static_cast<T>(softmax_value);
        }
    }
}

template <typename T>
__global__ void matrix_sigmoid(const T* a, T* b, const int rows, const int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        float val = static_cast<float>(a[index]);
        float sigmoid_val = 1.0f / (1.0f + expf(-val));
        b[index] = static_cast<T>(sigmoid_val);
    }
}

template <typename T>
__global__ void matrix_relu(const T *a, T* b, const int rows, const int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        if constexpr (std::is_same_v<T, __half>) {
            b[index] = __hmax(a[index], __half(0));
        } else {
            b[index] = max(T(0), a[index]);
        }
    }
}


template <typename T>
__global__ void matrix_relu_backward(const T* a, const T* grad_output, T* grad_input, const int rows, const int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        int index = row * cols + col;
        T zero = T(0);
        grad_input[index] = a[index] > zero ? grad_output[index] : zero;
    }
}

// Random numbers drawn from uniform distribution
template <typename T>
__global__ void fill_with_random(T *a, const unsigned long seed, const int rows, const int cols, const T min, const T max) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        curandState_t state;
        curand_init(seed, index, 0, &state);

        float rand_uniform = curand_uniform(&state);
        T val = static_cast<T>(rand_uniform);
        a[index] = min + val * (max - min);
    }
}

template <typename T>
__global__ void ce_loss(const T* preds, const T* labels, T* losses, const int rows, const int cols, const float epsilon) {
    int row = threadIdx.y;

    if (row < rows) {
        int label = (int)labels[row];
        float pred = static_cast<float>(preds[row * cols + label]);
        float val = -1 * logf(pred + epsilon);
        losses[row] = static_cast<T>(val);
    }
}

template <typename T>
__global__ void ce_loss_and_predictions(const T* preds, const T* labels, T* losses, T* correct_predictions, const int rows, const int cols, const float epsilon) {
    int row = threadIdx.y;

    if (row < rows) {
        int label = (int)labels[row];
        T pred = preds[row * cols + label];
        float float_pred = static_cast<float>(pred);
        float loss = -1 * logf(float_pred + epsilon);
        losses[row] = static_cast<T>(loss);

        int predicted_label = 0;
        T max_prob = preds[row * cols];
        for (int col = 1; col < cols; ++col) {
            T current_prob = preds[row * cols + col];
            if (current_prob > max_prob) {
                max_prob = current_prob;
                predicted_label = col;
            }
        }
        correct_predictions[row] = (predicted_label == label) ? 1.0f : 0.0f;
    }
}

template <typename T>
__global__ void softmax_bwd(const T* labels, const T* softmax_outputs, T* softmax_grads, const int rows, const int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        int label_idx = (int)labels[row];

        // https://shivammehta25.github.io/posts/deriving-categorical-cross-entropy-and-softmax/#derivation-of-softmax
        if (col == label_idx) {
            softmax_grads[idx] = softmax_outputs[idx] - T(1);
        } else {
            softmax_grads[idx] = softmax_outputs[idx];
        }
    }
}

__global__ void convertFP32ToFP16(half* out, float* in, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = __float2half(in[idx]);
    }
}

__global__ void convertFP16ToFP32(float* out, half* in, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = __half2float(in[idx]);
    }
}
