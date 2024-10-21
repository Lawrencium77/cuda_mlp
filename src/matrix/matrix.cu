// Implements Vector class

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
        std::cerr << "Softmax kernels doesn't support matrix width > 256" << std::endl;
        exit(1);
    }
    Matrix result(rows, cols);

    dim3 blockSize(1, 256);
    dim3 gridSize(1, (rows + blockSize.y - 1) / blockSize.y);

    matrix_softmax<<<blockSize, gridSize>>>(data, result.data, rows, cols);
    cudaDeviceSynchronize();
    return result;
};
