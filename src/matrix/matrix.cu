#include "matrix.h"
#include "matrix_kernels.h"
#include <iostream>

Matrix::Matrix() : rows(0), cols(0), numel(0), data(nullptr) {}

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols), numel(rows * cols) {
    cudaMalloc(&data, numel * sizeof(float));
}

Matrix::~Matrix() {
    cudaFree(data);
}

void Matrix::setData(const float* host_data) {
    cudaMemcpy(data, host_data, numel * sizeof(float), cudaMemcpyHostToDevice);
}

void Matrix::getData(float* host_data) const {
    cudaMemcpy(host_data, data, numel * sizeof(float), cudaMemcpyDeviceToHost);
}

Matrix::Matrix(const Matrix& other) : rows(other.rows), cols(other.cols), numel(other.numel) {
    cudaMalloc(&data, numel * sizeof(float));
    cudaMemcpy(data, other.data, numel * sizeof(float), cudaMemcpyDeviceToDevice);
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        cudaFree(data);

        rows = other.rows;
        cols = other.cols;
        numel = other.numel;

        cudaMalloc(&data, numel * sizeof(float));
        cudaMemcpy(data, other.data, numel * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    return *this;
}

float matabsmax(const Matrix& mat){
    float* d_max;
    cudaMalloc(&d_max, sizeof(float));
    cudaMemset(d_max, 0, sizeof(float));

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (mat.cols + blockSize.x - 1) / blockSize.x,
        (mat.rows + blockSize.y - 1) / blockSize.y
    );

    matrix_max_abs<<<gridSize, blockSize>>>(mat.data, d_max, mat.rows, mat.cols);
    cudaDeviceSynchronize();

    float h_sum = 0.0f;
    cudaMemcpy(&h_sum, d_max, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_max);
    return h_sum;
}

float matsum(const Matrix& mat){
    float* d_sum;
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (mat.cols + blockSize.x - 1) / blockSize.x,
        (mat.rows + blockSize.y - 1) / blockSize.y
    );

    matrix_sum<<<gridSize, blockSize>>>(mat.data, d_sum, mat.rows, mat.cols);
    cudaDeviceSynchronize();

    float h_sum = 0.0f;
    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_sum);
    return h_sum;
}

Matrix transpose(const Matrix& mat) {
    Matrix result(mat.cols, mat.rows);

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (mat.cols + blockSize.x - 1) / blockSize.x,
        (mat.rows + blockSize.y - 1) / blockSize.y
    );

    matrix_transpose<<<gridSize, blockSize>>>(mat.data, result.data, mat.rows, mat.cols);
    cudaDeviceSynchronize();
    return result;
}

Matrix softmax(const Matrix& mat) {
    Matrix result(mat.rows, mat.cols);

    dim3 blockSize(1, 1024);
    dim3 gridSize(1, (mat.rows + 1024 - 1) / 1024);

    matrix_softmax_over_rows<<<gridSize, blockSize>>>(mat.data, result.data, mat.rows, mat.cols);
    cudaDeviceSynchronize();
    return result;
};

Matrix sigmoid(const Matrix& mat) {
    Matrix result(mat.rows, mat.cols);

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (mat.cols + blockSize.x - 1) / blockSize.x,
        (mat.rows + blockSize.y - 1) / blockSize.y
    );

    matrix_sigmoid<<<gridSize, blockSize>>>(mat.data, result.data, mat.rows, mat.cols);
    cudaDeviceSynchronize();
    return result;
};

Matrix relu(const Matrix& mat) {
    Matrix result(mat.rows, mat.cols);

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (mat.cols + blockSize.x - 1) / blockSize.x,
        (mat.rows + blockSize.y - 1) / blockSize.y
    );

    matrix_relu<<<gridSize, blockSize>>>(mat.data, result.data, mat.rows, mat.cols);
    cudaDeviceSynchronize();
    return result;
}

Matrix operator+(const Matrix& mat, const float value) {
    Matrix result(mat.rows, mat.cols);

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (mat.cols - 1) / blockSize.x + 1,
        (mat.rows - 1) / blockSize.y + 1
    );

    matrix_const_add<<<gridSize, blockSize>>>(mat.data, value, result.data, mat.rows, mat.cols);
    cudaDeviceSynchronize();
    return result;
}

Matrix operator*(const Matrix& mat, const float value) {
    Matrix result(mat.rows, mat.cols);

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (mat.cols - 1) / blockSize.x + 1,
        (mat.rows - 1) / blockSize.y + 1
    );

    matrix_const_mul<<<gridSize, blockSize>>>(mat.data, value, result.data, mat.rows, mat.cols);
    cudaDeviceSynchronize();
    return result;
}

Matrix operator-(const float value, const Matrix& mat) {
    Matrix negative_matrix = mat * -1.0f;
    return negative_matrix + value;
}

Matrix operator/(const Matrix& mat, const float value) {
    float inv_value = 1 / value;
    return mat * inv_value;
}

Matrix operator+(const Matrix& mat1, const Matrix& mat2) {
    if (mat1.rows != mat2.rows || mat1.cols != mat2.cols){
        std::cerr << "Matrix dimensions must match for addition!" << std::endl;
        exit(1);
    }
    Matrix result(mat1.rows, mat1.cols);

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (mat1.cols - 1) / blockSize.x + 1,
        (mat1.rows - 1) / blockSize.y + 1
    );

    matrix_add<<<gridSize, blockSize>>>(mat1.data, mat2.data, result.data, mat1.rows, mat1.cols);
    cudaDeviceSynchronize();
    return result;
}

Matrix operator*(const Matrix& mat1, const Matrix& mat2) {
    if (mat1.rows != mat2.rows || mat1.cols != mat2.cols){
        std::cerr << "Matrix dimensions must match for Hadamard product!" << std::endl;
        exit(1);
    }
    Matrix result(mat1.rows, mat1.cols);

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (mat1.cols - 1) / blockSize.x + 1,
        (mat1.rows - 1) / blockSize.y + 1
    );

    matrix_hadamard<<<gridSize, blockSize>>>(mat1.data, mat2.data, result.data, mat1.rows, mat1.cols);
    cudaDeviceSynchronize();
    return result;
}

Matrix matmul(const Matrix& mat1, const Matrix& mat2) {
    if (mat1.cols != mat2.rows){
        std::cerr << "Trying to multiply two matrices with non-matchiing inner dim" << std::endl;
        exit(1);
    }

    Matrix result(mat1.rows, mat2.cols);

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (mat2.cols - 1) / blockSize.x + 1,
        (mat1.rows - 1) / blockSize.y + 1
    );

    matrix_multiply<<<gridSize, blockSize>>>(mat1.data, mat2.data, result.data, mat1.rows, mat1.cols, mat2.cols);
    cudaDeviceSynchronize();
    return result;
};

Matrix relu_backward(const Matrix& mat1, const Matrix& grad_output) {
    Matrix grad_input(mat1.rows, mat1.cols);

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (mat1.cols + blockSize.x - 1) / blockSize.x,
        (mat1.rows + blockSize.y - 1) / blockSize.y
    );

    matrix_relu_backward<<<gridSize, blockSize>>>(mat1.data, grad_output.data, grad_input.data, mat1.rows, mat1.cols);
    cudaDeviceSynchronize();
    return grad_input;
}

void Matrix::random(const unsigned long seed, const float min, const float max) {
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (cols + blockSize.x - 1) / blockSize.x,
        (rows + blockSize.y - 1) / blockSize.y
    );

    fill_with_random<<<gridSize, blockSize>>>(data, seed, rows, cols, min, max);
    cudaDeviceSynchronize();
};

Matrix get_ce_loss(const Matrix& mat1, const Matrix& labels) {
    if (mat1.rows != labels.rows) {
        std::cerr << "Non-matching number of rows for input and labels" << std::endl;
        exit(1);
    }

    Matrix losses = Matrix(mat1.rows, 1);

    dim3 blockSize(1, 1024);
    dim3 gridSize(1, 1);

    ce_loss<<<gridSize, blockSize>>>(mat1.data, labels.data, losses.data, mat1.rows, mat1.cols);
    cudaDeviceSynchronize();
    return losses;
};

//  labels => (bsz, 1) => represents the index of the correct output
//  softmax_output => (bsz, num_classes)
Matrix ce_softmax_bwd(const Matrix& labels, const Matrix& softmax_output) {
    int bsz = softmax_output.rows;
    int num_classes = softmax_output.cols;

    if (labels.rows != bsz) {
        std::cerr << "Non-matching number of rows for input and labels" << std::endl;
        exit(1);
    }

    Matrix softmax_grads = Matrix(bsz, num_classes);

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (num_classes + blockSize.x - 1) / blockSize.x,
        (bsz + blockSize.y - 1) / blockSize.y
    );

    softmax_bwd<<<gridSize, blockSize>>>(labels.data, softmax_output.data, softmax_grads.data, bsz, num_classes);
    cudaDeviceSynchronize();
    return softmax_grads;
}
