#include "matrix.h"
#include "matrix_kernels.h"
#include <iostream>

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols), numel(rows * cols) {
    cudaMalloc(&data, numel * sizeof(float));
}

Matrix::~Matrix() {
    cudaFree(data);
}

void Matrix::setData(const float* host_data) {
    cudaMemcpy(data, host_data, numel * sizeof(float), cudaMemcpyHostToDevice);
}

void Matrix::getData(float* host_data) {
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

Matrix Matrix::operator+(const float value) {
    Matrix result(rows, cols);

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (cols - 1) / blockSize.x + 1, // Ceil(cols / blockSize.x)
        (rows - 1) / blockSize.y + 1 // Ceil(rows / blockSize.y)
    );

    matrix_const_add<<<gridSize, blockSize>>>(data, value, result.data, rows, cols);
    cudaDeviceSynchronize();
    return result;
}

Matrix Matrix::operator-(const float value) {
    float negative_value = -value;
    return *this + negative_value;
}

Matrix Matrix::operator*(const float value) {
    Matrix result(rows, cols);

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (cols - 1) / blockSize.x + 1, // Ceil(cols / blockSize.x)
        (rows - 1) / blockSize.y + 1 // Ceil(rows / blockSize.y)
    );

    matrix_const_mul<<<gridSize, blockSize>>>(data, value, result.data, rows, cols);
    cudaDeviceSynchronize();
    return result;
}

Matrix Matrix::operator/(const float value) {
    float inv_value = 1 / value;
    return *this * inv_value;
}

Matrix Matrix::operator+(Matrix& other) {
    if (rows != other.rows || cols != other.cols){
        std::cerr << "Matrix dimensions must match for addition!" << std::endl;
        exit(1);
    }
    Matrix result(rows, cols);

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (cols - 1) / blockSize.x + 1, // Ceil(cols / blockSize.x)
        (rows - 1) / blockSize.y + 1 // Ceil(rows / blockSize.y)
    );

    matrix_add<<<gridSize, blockSize>>>(data, other.data, result.data, rows, cols);
    cudaDeviceSynchronize();
    return result;
}

Matrix Matrix::operator*(Matrix& other) {
    if (rows != other.rows || cols != other.cols){
        std::cerr << "Matrix dimensions must match for Hadamard product!" << std::endl;
        exit(1);
    }
    Matrix result(rows, cols);

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (cols - 1) / blockSize.x + 1, // Ceil(cols / blockSize.x)
        (rows - 1) / blockSize.y + 1 // Ceil(rows / blockSize.y)
    );

    matrix_hadamard<<<gridSize, blockSize>>>(data, other.data, result.data, rows, cols);
    cudaDeviceSynchronize();
    return result;
}

Matrix Matrix::transpose() {
    Matrix result(cols, rows);

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (cols + blockSize.x - 1) / blockSize.x,  // Ceil(cols / blockSize.x)
        (rows + blockSize.y - 1) / blockSize.y   // Ceil(rows / blockSize.y)
    );

    matrix_transpose<<<gridSize, blockSize>>>(data, result.data, rows, cols);
    cudaDeviceSynchronize();
    return result;
}

Matrix Matrix::matmul(const Matrix& other) {
    if (cols != other.rows){
        std::cerr << "Trying to multiply two matrices with non-matchiing inner dim" << std::endl;
        exit(1);
    }

    Matrix result(rows, other.cols);

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (other.cols - 1) / blockSize.x + 1, // Ceil(cols / blockSize.x)
        (rows - 1) / blockSize.y + 1 // Ceil(rows / blockSize.y)
    );

    matrix_multiply<<<gridSize, blockSize>>>(data, other.data, result.data, rows, cols, other.cols);
    cudaDeviceSynchronize();
    return result;
};

Matrix Matrix::softmax() {
    if (cols > 256){
        std::cerr << "Softmax kernel doesn't support matrix width > 256" << std::endl;
        exit(1);
    }
    Matrix result(rows, cols);

    dim3 blockSize(1, 256);
    dim3 gridSize(cols, 1);

    matrix_softmax<<<blockSize, gridSize>>>(data, result.data, rows, cols);
    cudaDeviceSynchronize();
    return result;
};

Matrix Matrix::sigmoid() {
    Matrix result(rows, cols);

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (cols + blockSize.x - 1) / blockSize.x,
        (rows + blockSize.y - 1) / blockSize.y
    );

    matrix_sigmoid<<<gridSize, blockSize>>>(data, result.data, rows, cols);
    cudaDeviceSynchronize();
    return result;
};

void Matrix::random(unsigned long seed) {
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (cols + blockSize.x - 1) / blockSize.x,
        (rows + blockSize.y - 1) / blockSize.y
    );

    fill_with_random<<<gridSize, blockSize>>>(data, seed, rows, cols);
    cudaDeviceSynchronize();
};

Matrix Matrix::get_ce_loss(Matrix& labels) {
    if (cols != labels.cols) {
        std::cerr << "Non-matching number of columns for input and labels" << std::endl;
        exit(1);
    }

    Matrix losses = Matrix(1, cols);

    dim3 blockSize(1, 32);
    dim3 gridSize(1, cols / blockSize.y + 1);

    ce_loss<<<gridSize, blockSize>>>(data, labels.data, losses.data, rows, cols);
    cudaDeviceSynchronize();
    return losses;
};
