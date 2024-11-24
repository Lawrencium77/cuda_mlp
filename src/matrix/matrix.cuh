MemoryAllocator baseMatrix::allocator;

template <typename T>
Matrix<T>::Matrix()
    : rows(0), cols(0), numel(0), host_data(nullptr), device_data(nullptr) {}

template <typename T>
Matrix<T>::Matrix(int rows, int cols)
    : rows(rows), cols(cols), numel(rows * cols) {
  host_data = new float[numel];
  device_data = static_cast<T *>(allocator.allocate(numel * sizeof(T)));
}

template <typename T> Matrix<T>::~Matrix() {
  delete[] host_data;
  allocator.free(device_data);
}

template <typename T>
void Matrix<T>::toDevice() {
    if constexpr (std::is_same_v<T, half>) {
        // FP32 -> FP16 conversion
        float* temp_device_data;
        temp_device_data = static_cast<float *>(allocator.allocate(numel * sizeof(float)));
        cudaMemcpy(temp_device_data, host_data, numel * sizeof(float), cudaMemcpyHostToDevice);

        int blockSize = 256;
        int numBlocks = (numel + blockSize - 1) / blockSize;
        convertFP32ToFP16<<<numBlocks, blockSize>>>(device_data, temp_device_data, numel);

        allocator.free(temp_device_data);
    } else {
        cudaMemcpy(device_data, host_data, numel * sizeof(float),
                   cudaMemcpyHostToDevice);
    }
    CHECK_CUDA_STATE();
}

template <typename T> void Matrix<T>::toHost() {
  if constexpr (std::is_same_v<T, half>) {
        // FP16 -> FP32 conversion
        float* temp_device_data;
        temp_device_data = static_cast<float *>(allocator.allocate(numel * sizeof(float)));

        int blockSize = 256;
        int numBlocks = (numel + blockSize - 1) / blockSize;
        convertFP16ToFP32<<<numBlocks, blockSize>>>(temp_device_data, device_data, numel);

        cudaMemcpy(host_data, temp_device_data, numel * sizeof(float), cudaMemcpyDeviceToHost);

        allocator.free(temp_device_data);
    } else {
        cudaMemcpy(host_data, device_data, numel * sizeof(float), cudaMemcpyDeviceToHost);
    }
    CHECK_CUDA_STATE();
}

template <typename T> void Matrix<T>::setHostData(float *data) {
  delete[] host_data;
  host_data = data;
}

template <typename T>
Matrix<T>::Matrix(Matrix<T> &&other)
    : rows(other.rows), cols(other.cols), numel(other.numel),
      host_data(other.host_data), device_data(other.device_data) {
  other.rows = 0;
  other.cols = 0;
  other.numel = 0;
  other.host_data = nullptr;
  other.device_data = nullptr;
}

template <typename T> Matrix<T> &Matrix<T>::operator=(Matrix<T> &&other) {
  if (this != &other) {
    delete[] host_data;
    allocator.free(device_data);

    rows = other.rows;
    cols = other.cols;
    numel = other.numel;
    host_data = other.host_data;
    device_data = other.device_data;

    other.rows = 0;
    other.cols = 0;
    other.numel = 0;
    other.host_data = nullptr;
    other.device_data = nullptr;
  }
  return *this;
}

template <typename T> Matrix<T> &Matrix<T>::operator=(const Matrix<T> &other) {
  if (this != &other) {
    // Deallocate and reallocate resources since we can't assume numel ==
    // other.numel
    delete[] host_data;
    allocator.free(device_data);

    rows = other.rows;
    cols = other.cols;
    numel = other.numel;

    host_data = new float[numel];
    std::copy(other.host_data, other.host_data + numel, host_data);

    device_data =
        static_cast<float *>(allocator.allocate(numel * sizeof(float)));
    cudaMemcpy(device_data, other.device_data, numel * sizeof(float),
               cudaMemcpyDeviceToDevice);
  }
  return *this;
}

template <typename T> void Matrix<T>::printData(std::string message) {
  toHost();
  if (message != "") {
    std::cout << message << ": \n";
  }
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      std::cout << host_data[i * cols + j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

template <typename T> float matsum(const Matrix<T> &mat) {
  float *d_sum;
  cudaError_t malloc_err = cudaMalloc(&d_sum, sizeof(float));
  cudaError_t memset_err = cudaMemset(d_sum, 0, sizeof(float));
  CHECK_CUDA_STATE_WITH_ERR(malloc_err);
  CHECK_CUDA_STATE_WITH_ERR(memset_err);

  dim3 blockSize(16, 16);
  dim3 gridSize((mat.cols + blockSize.x - 1) / blockSize.x,
                (mat.rows + blockSize.y - 1) / blockSize.y);

  matrix_sum<T>
      <<<gridSize, blockSize>>>(mat.device_data, d_sum, mat.rows, mat.cols);
  cudaDeviceSynchronize();
  CHECK_CUDA_STATE();

  float h_sum = 0.0f;
  cudaError_t memcpy_err =
      cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
  CHECK_CUDA_STATE_WITH_ERR(memcpy_err);

  cudaError_t free_err = cudaFree(d_sum);
  CHECK_CUDA_STATE_WITH_ERR(free_err);
  return h_sum;
}

template <typename T> Matrix<T> transpose(const Matrix<T> &mat) {
  Matrix<T> result(mat.cols, mat.rows);

  dim3 blockSize(16, 16);
  dim3 gridSize((mat.cols + blockSize.x - 1) / blockSize.x,
                (mat.rows + blockSize.y - 1) / blockSize.y);

  matrix_transpose<T><<<gridSize, blockSize>>>(
      mat.device_data, result.device_data, mat.rows, mat.cols);
  CHECK_CUDA_STATE();
  return result;
}

template <typename T> Matrix<T> softmax(const Matrix<T> &mat) {
  Matrix<T> result(mat.rows, mat.cols);

  dim3 blockSize(1, 1024);
  dim3 gridSize(1, (mat.rows + 1024 - 1) / 1024);

  matrix_softmax_over_rows<T><<<gridSize, blockSize>>>(
      mat.device_data, result.device_data, mat.rows, mat.cols);
  CHECK_CUDA_STATE();
  return result;
};

template <typename T> Matrix<T> sigmoid(const Matrix<T> &mat) {
  Matrix<T> result(mat.rows, mat.cols);

  dim3 blockSize(16, 16);
  dim3 gridSize((mat.cols + blockSize.x - 1) / blockSize.x,
                (mat.rows + blockSize.y - 1) / blockSize.y);

  matrix_sigmoid<T><<<gridSize, blockSize>>>(
      mat.device_data, result.device_data, mat.rows, mat.cols);
  CHECK_CUDA_STATE();
  return result;
};

template <typename T> Matrix<T> relu(const Matrix<T> &mat) {
  Matrix<T> result(mat.rows, mat.cols);

  dim3 blockSize(16, 16);
  dim3 gridSize((mat.cols + blockSize.x - 1) / blockSize.x,
                (mat.rows + blockSize.y - 1) / blockSize.y);

  matrix_relu<T><<<gridSize, blockSize>>>(mat.device_data, result.device_data,
                                          mat.rows, mat.cols);
  CHECK_CUDA_STATE();
  return result;
}

template <typename T>
Matrix<T> operator+(const Matrix<T> &mat, const float value) {
  Matrix<T> result(mat.rows, mat.cols);

  dim3 blockSize(16, 16);
  dim3 gridSize((mat.cols - 1) / blockSize.x + 1,
                (mat.rows - 1) / blockSize.y + 1);

  matrix_const_add<T><<<gridSize, blockSize>>>(
      mat.device_data, value, result.device_data, mat.rows, mat.cols);
  CHECK_CUDA_STATE();
  return result;
}

template <typename T>
Matrix<T> operator*(const Matrix<T> &mat, const float value) {
  Matrix<T> result(mat.rows, mat.cols);

  dim3 blockSize(16, 16);
  dim3 gridSize((mat.cols - 1) / blockSize.x + 1,
                (mat.rows - 1) / blockSize.y + 1);

  matrix_const_mul<T><<<gridSize, blockSize>>>(
      mat.device_data, value, result.device_data, mat.rows, mat.cols);
  CHECK_CUDA_STATE();
  return result;
}

template <typename T>
Matrix<T> operator-(const float value, const Matrix<T> &mat) {
  Matrix<T> negative_matrix = mat * -1.0f;
  return negative_matrix + value;
}

template <typename T>
Matrix<T> operator/(const Matrix<T> &mat, const float value) {
  float inv_value = 1 / value;
  return mat * inv_value;
}

template <typename T>
Matrix<T> operator+(const Matrix<T> &mat1, const Matrix<T> &mat2) {
  if (mat1.rows != mat2.rows || mat1.cols != mat2.cols) {
    throw std::runtime_error("Matrix dimensions must match for addition");
  }
  Matrix<T> result(mat1.rows, mat1.cols);

  dim3 blockSize(16, 16);
  dim3 gridSize((mat1.cols - 1) / blockSize.x + 1,
                (mat1.rows - 1) / blockSize.y + 1);

  matrix_add<T><<<gridSize, blockSize>>>(mat1.device_data, mat2.device_data,
                                         result.device_data, mat1.rows,
                                         mat1.cols);
  CHECK_CUDA_STATE();
  return result;
}

template <typename T>
Matrix<T> operator*(const Matrix<T> &mat1, const Matrix<T> &mat2) {
  if (mat1.rows != mat2.rows || mat1.cols != mat2.cols) {
    throw std::runtime_error(
        "Matrix dimensions must match for Hadamard product");
  }
  Matrix<T> result(mat1.rows, mat1.cols);

  dim3 blockSize(16, 16);
  dim3 gridSize((mat1.cols - 1) / blockSize.x + 1,
                (mat1.rows - 1) / blockSize.y + 1);

  matrix_hadamard<T>
      <<<gridSize, blockSize>>>(mat1.device_data, mat2.device_data,
                                result.device_data, mat1.rows, mat1.cols);
  CHECK_CUDA_STATE();
  return result;
}

template <typename T>
Matrix<T> matmul(const Matrix<T> &mat1, const Matrix<T> &mat2) {
  if (mat1.cols != mat2.rows) {
    throw std::runtime_error(
        "Trying to multiply two matrices with non-matchiing inner dim");
  }

  Matrix<T> result(mat1.rows, mat2.cols);

  dim3 blockSize(16, 16);
  dim3 gridSize((mat2.cols - 1) / blockSize.x + 1,
                (mat1.rows - 1) / blockSize.y + 1);

  matrix_multiply<T><<<gridSize, blockSize>>>(
      mat1.device_data, mat2.device_data, result.device_data, mat1.rows,
      mat1.cols, mat2.cols);
  CHECK_CUDA_STATE();
  return result;
};

template <typename T>
Matrix<T> relu_backward(const Matrix<T> &mat1, const Matrix<T> &grad_output) {
  Matrix<T> grad_input(mat1.rows, mat1.cols);

  dim3 blockSize(16, 16);
  dim3 gridSize((mat1.cols + blockSize.x - 1) / blockSize.x,
                (mat1.rows + blockSize.y - 1) / blockSize.y);

  matrix_relu_backward<T>
      <<<gridSize, blockSize>>>(mat1.device_data, grad_output.device_data,
                                grad_input.device_data, mat1.rows, mat1.cols);
  CHECK_CUDA_STATE();
  return grad_input;
}

template <typename T>
void Matrix<T>::random(const unsigned long seed, const T min, const T max) {
  dim3 blockSize(16, 16);
  dim3 gridSize((cols + blockSize.x - 1) / blockSize.x,
                (rows + blockSize.y - 1) / blockSize.y);

  fill_with_random<T>
      <<<gridSize, blockSize>>>(device_data, seed, rows, cols, min, max);
  CHECK_CUDA_STATE();
};

template <typename T>
Matrix<T> get_ce_loss(const Matrix<T> &mat1, const Matrix<T> &labels) {
  if (mat1.rows != labels.rows) {
    throw std::runtime_error(
        "Non-matching number of rows for input and labels");
  }

  Matrix<T> losses = Matrix<T>(mat1.rows, 1);

  dim3 blockSize(1, 1024);
  dim3 gridSize(1, 1);

  ce_loss<T><<<gridSize, blockSize>>>(mat1.device_data, labels.device_data,
                                      losses.device_data, mat1.rows, mat1.cols);
  CHECK_CUDA_STATE();
  return losses;
};

//  labels => (bsz, 1) => represents the index of the correct output
//  softmax_output => (bsz, num_classes)
template <typename T>
Matrix<T> ce_softmax_bwd(const Matrix<T> &labels,
                         const Matrix<T> &softmax_output) {
  int bsz = softmax_output.rows;
  int num_classes = softmax_output.cols;

  if (labels.rows != bsz) {
    throw std::runtime_error(
        "Non-matching number of rows for input and labels");
  }

  Matrix<T> softmax_grads = Matrix<T>(bsz, num_classes);

  dim3 blockSize(16, 16);
  dim3 gridSize((num_classes + blockSize.x - 1) / blockSize.x,
                (bsz + blockSize.y - 1) / blockSize.y);

  softmax_bwd<T>
      <<<gridSize, blockSize>>>(labels.device_data, softmax_output.device_data,
                                softmax_grads.device_data, bsz, num_classes);
  CHECK_CUDA_STATE();
  return softmax_grads;
}

template <typename T>
std::pair<Matrix<T>, Matrix<T>>
get_ce_loss_and_accuracy(const Matrix<T> &mat1, const Matrix<T> &labels) {
  if (mat1.rows != labels.rows) {
    throw std::runtime_error(
        "Non-matching number of rows for input and labels");
  }

  Matrix<T> losses = Matrix<T>(mat1.rows, 1);
  Matrix<T> predictions = Matrix<T>(mat1.rows, 1);

  dim3 blockSize(1, 1024);
  dim3 gridSize(1, 1);

  ce_loss_and_predictions<T><<<gridSize, blockSize>>>(
      mat1.device_data, labels.device_data, losses.device_data,
      predictions.device_data, mat1.rows, mat1.cols);
  CHECK_CUDA_STATE();
  return std::make_pair(std::move(losses), std::move(predictions));
};
