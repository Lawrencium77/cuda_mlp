#include "matrix.h"
#include "matrix_kernels.h"
#include <iostream>

// Allocator Setup
std::unique_ptr<AllocatorBase> Matrix::allocator;

static std::unique_ptr<AllocatorBase> createAllocator() {
  const char *env = std::getenv("ALLOCATOR_TYPE");
  if (env && std::string(env) == "cuda") {
    std::cerr << "[INFO] Using CudaAsyncAllocator\n";
    return std::make_unique<CudaAsyncAllocator>();
  } else {
    std::cerr << "[INFO] Using MemoryAllocator (default)\n";
    return std::make_unique<MemoryAllocator>();
  }
}

__attribute__((constructor)) static void initAllocator() {
  Matrix::allocator = createAllocator();
}

// Rest of functionality
Matrix::Matrix()
    : rows(0), cols(0), numel(0), host_data(nullptr), device_data(nullptr) {}

Matrix::Matrix(int rows, int cols)
    : rows(rows), cols(cols), numel(rows * cols) {
  host_data = new float[numel];
  device_data =
      static_cast<float *>(allocator->allocate(numel * sizeof(float)));
}

Matrix::~Matrix() {
  delete[] host_data;
  allocator->free(device_data);
}

void Matrix::toDevice() {
  cudaMemcpy(device_data, host_data, numel * sizeof(float),
             cudaMemcpyHostToDevice);
}

void Matrix::toHost() {
  cudaMemcpy(host_data, device_data, numel * sizeof(float),
             cudaMemcpyDeviceToHost);
}

void Matrix::setHostData(float *data) {
  delete[] host_data;
  host_data = data;
}

Matrix::Matrix(Matrix &&other)
    : rows(other.rows), cols(other.cols), numel(other.numel),
      host_data(other.host_data), device_data(other.device_data) {
  other.rows = 0;
  other.cols = 0;
  other.numel = 0;
  other.host_data = nullptr;
  other.device_data = nullptr;
}

Matrix &Matrix::operator=(Matrix &&other) {
  if (this != &other) {
    delete[] host_data;
    allocator->free(device_data);

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

Matrix &Matrix::operator=(const Matrix &other) {
  if (this != &other) {
    // Deallocate and reallocate resources since we can't assume numel ==
    // other.numel
    delete[] host_data;
    allocator->free(device_data);

    rows = other.rows;
    cols = other.cols;
    numel = other.numel;

    host_data = new float[numel];
    std::copy(other.host_data, other.host_data + numel, host_data);

    device_data =
        static_cast<float *>(allocator->allocate(numel * sizeof(float)));
    cudaMemcpy(device_data, other.device_data, numel * sizeof(float),
               cudaMemcpyDeviceToDevice);
  }
  return *this;
}

void Matrix::printData(std::string message) {
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

float matsum(const Matrix &mat) {
  float *d_sum;
  d_sum = static_cast<float *>(Matrix::allocator->allocate(sizeof(float)));
  cudaError_t memset_err = cudaMemsetAsync(d_sum, 0, sizeof(float));
  CHECK_CUDA_STATE_WITH_ERR(memset_err);

  dim3 blockSize(16, 16);
  dim3 gridSize((mat.cols + blockSize.x - 1) / blockSize.x,
                (mat.rows + blockSize.y - 1) / blockSize.y);

  matrix_sum<<<gridSize, blockSize>>>(mat.device_data, d_sum, mat.rows,
                                      mat.cols);
  cudaDeviceSynchronize();
  CHECK_CUDA_STATE();

  float h_sum = 0.0f;
  cudaError_t memcpy_err =
      cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
  CHECK_CUDA_STATE_WITH_ERR(memcpy_err);

  Matrix::allocator->free(d_sum);
  cudaDeviceSynchronize();
  return h_sum;
}

Matrix transpose(const Matrix &mat) {
  Matrix result(mat.cols, mat.rows);

  dim3 blockSize(16, 16);
  dim3 gridSize((mat.cols + blockSize.x - 1) / blockSize.x,
                (mat.rows + blockSize.y - 1) / blockSize.y);

  matrix_transpose<<<gridSize, blockSize>>>(mat.device_data, result.device_data,
                                            mat.rows, mat.cols);
  CHECK_CUDA_STATE();
  return result;
}

Matrix softmax(const Matrix &mat) {
  Matrix result(mat.rows, mat.cols);

  dim3 blockSize(1, 1024);
  dim3 gridSize(1, (mat.rows + 1024 - 1) / 1024);

  matrix_softmax_over_rows<<<gridSize, blockSize>>>(
      mat.device_data, result.device_data, mat.rows, mat.cols);
  CHECK_CUDA_STATE();
  return result;
};

Matrix sigmoid(const Matrix &mat) {
  Matrix result(mat.rows, mat.cols);

  dim3 blockSize(16, 16);
  dim3 gridSize((mat.cols + blockSize.x - 1) / blockSize.x,
                (mat.rows + blockSize.y - 1) / blockSize.y);

  matrix_sigmoid<<<gridSize, blockSize>>>(mat.device_data, result.device_data,
                                          mat.rows, mat.cols);
  CHECK_CUDA_STATE();
  return result;
};

Matrix relu(const Matrix &mat) {
  Matrix result(mat.rows, mat.cols);

  dim3 blockSize(16, 16);
  dim3 gridSize((mat.cols + blockSize.x - 1) / blockSize.x,
                (mat.rows + blockSize.y - 1) / blockSize.y);

  matrix_relu<<<gridSize, blockSize>>>(mat.device_data, result.device_data,
                                       mat.rows, mat.cols);
  CHECK_CUDA_STATE();
  return result;
}

Matrix operator+(const Matrix &mat, const float value) {
  Matrix result(mat.rows, mat.cols);

  dim3 blockSize(16, 16);
  dim3 gridSize((mat.cols - 1) / blockSize.x + 1,
                (mat.rows - 1) / blockSize.y + 1);

  matrix_const_add<<<gridSize, blockSize>>>(
      mat.device_data, value, result.device_data, mat.rows, mat.cols);
  CHECK_CUDA_STATE();
  return result;
}

Matrix operator*(const Matrix &mat, const float value) {
  Matrix result(mat.rows, mat.cols);

  dim3 blockSize(16, 16);
  dim3 gridSize((mat.cols - 1) / blockSize.x + 1,
                (mat.rows - 1) / blockSize.y + 1);

  matrix_const_mul<<<gridSize, blockSize>>>(
      mat.device_data, value, result.device_data, mat.rows, mat.cols);
  CHECK_CUDA_STATE();
  return result;
}

Matrix operator-(const float value, const Matrix &mat) {
  Matrix negative_matrix = mat * -1.0f;
  return negative_matrix + value;
}

Matrix operator/(const Matrix &mat, const float value) {
  float inv_value = 1 / value;
  return mat * inv_value;
}

Matrix operator+(const Matrix &mat1, const Matrix &mat2) {
  if (mat1.rows != mat2.rows || mat1.cols != mat2.cols) {
    throw std::runtime_error("Matrix dimensions must match for addition");
  }
  Matrix result(mat1.rows, mat1.cols);

  dim3 blockSize(16, 16);
  dim3 gridSize((mat1.cols - 1) / blockSize.x + 1,
                (mat1.rows - 1) / blockSize.y + 1);

  matrix_add<<<gridSize, blockSize>>>(mat1.device_data, mat2.device_data,
                                      result.device_data, mat1.rows, mat1.cols);
  CHECK_CUDA_STATE();
  return result;
}

Matrix operator*(const Matrix &mat1, const Matrix &mat2) {
  if (mat1.rows != mat2.rows || mat1.cols != mat2.cols) {
    throw std::runtime_error(
        "Matrix dimensions must match for Hadamard product");
  }
  Matrix result(mat1.rows, mat1.cols);

  dim3 blockSize(16, 16);
  dim3 gridSize((mat1.cols - 1) / blockSize.x + 1,
                (mat1.rows - 1) / blockSize.y + 1);

  matrix_hadamard<<<gridSize, blockSize>>>(mat1.device_data, mat2.device_data,
                                           result.device_data, mat1.rows,
                                           mat1.cols);
  CHECK_CUDA_STATE();
  return result;
}

Matrix matmul(const Matrix &mat1, const Matrix &mat2) {
  if (mat1.cols != mat2.rows) {
    throw std::runtime_error(
        "Trying to multiply two matrices with non-matchiing inner dim");
  }

  Matrix result(mat1.rows, mat2.cols);

  dim3 blockSize(16, 16);
  dim3 gridSize((mat2.cols - 1) / blockSize.x + 1,
                (mat1.rows - 1) / blockSize.y + 1);

  matrix_multiply<<<gridSize, blockSize>>>(mat1.device_data, mat2.device_data,
                                           result.device_data, mat1.rows,
                                           mat1.cols, mat2.cols);
  CHECK_CUDA_STATE();
  return result;
};

Matrix relu_backward(const Matrix &mat1, const Matrix &grad_output) {
  Matrix grad_input(mat1.rows, mat1.cols);

  dim3 blockSize(16, 16);
  dim3 gridSize((mat1.cols + blockSize.x - 1) / blockSize.x,
                (mat1.rows + blockSize.y - 1) / blockSize.y);

  matrix_relu_backward<<<gridSize, blockSize>>>(
      mat1.device_data, grad_output.device_data, grad_input.device_data,
      mat1.rows, mat1.cols);
  CHECK_CUDA_STATE();
  return grad_input;
}

void Matrix::random(const unsigned long seed, const float min,
                    const float max) {
  dim3 blockSize(16, 16);
  dim3 gridSize((cols + blockSize.x - 1) / blockSize.x,
                (rows + blockSize.y - 1) / blockSize.y);

  fill_with_random<<<gridSize, blockSize>>>(device_data, seed, rows, cols, min,
                                            max);
  CHECK_CUDA_STATE();
};

Matrix get_ce_loss(const Matrix &mat1, const Matrix &labels) {
  if (mat1.rows != labels.rows) {
    throw std::runtime_error(
        "Non-matching number of rows for input and labels");
  }

  Matrix losses = Matrix(mat1.rows, 1);

  dim3 blockSize(1, 1024);
  dim3 gridSize(1, 1);

  ce_loss<<<gridSize, blockSize>>>(mat1.device_data, labels.device_data,
                                   losses.device_data, mat1.rows, mat1.cols);
  CHECK_CUDA_STATE();
  return losses;
};

//  labels => (bsz, 1) => represents the index of the correct output
//  softmax_output => (bsz, num_classes)
Matrix ce_softmax_bwd(const Matrix &labels, const Matrix &softmax_output) {
  int bsz = softmax_output.rows;
  int num_classes = softmax_output.cols;

  if (labels.rows != bsz) {
    throw std::runtime_error(
        "Non-matching number of rows for input and labels");
  }

  Matrix softmax_grads = Matrix(bsz, num_classes);

  dim3 blockSize(16, 16);
  dim3 gridSize((num_classes + blockSize.x - 1) / blockSize.x,
                (bsz + blockSize.y - 1) / blockSize.y);

  softmax_bwd<<<gridSize, blockSize>>>(
      labels.device_data, softmax_output.device_data, softmax_grads.device_data,
      bsz, num_classes);
  CHECK_CUDA_STATE();
  return softmax_grads;
}

std::pair<Matrix, Matrix> get_ce_loss_and_accuracy(const Matrix &mat1,
                                                   const Matrix &labels) {
  if (mat1.rows != labels.rows) {
    throw std::runtime_error(
        "Non-matching number of rows for input and labels");
  }

  Matrix losses = Matrix(mat1.rows, 1);
  Matrix predictions = Matrix(mat1.rows, 1);

  dim3 blockSize(1, 1024);
  dim3 gridSize(1, 1);

  ce_loss_and_predictions<<<gridSize, blockSize>>>(
      mat1.device_data, labels.device_data, losses.device_data,
      predictions.device_data, mat1.rows, mat1.cols);
  CHECK_CUDA_STATE();
  return std::make_pair(std::move(losses), std::move(predictions));
};
