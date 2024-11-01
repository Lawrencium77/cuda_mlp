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
        (cols - 1) / blockSize.x + 1,
        (rows - 1) / blockSize.y + 1
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
        (cols - 1) / blockSize.x + 1,
        (rows - 1) / blockSize.y + 1
    );

    matrix_const_mul<<<gridSize, blockSize>>>(data, value, result.data, rows, cols);
    cudaDeviceSynchronize();
    return result;
}

Matrix Matrix::operator/(const float value) {
    float inv_value = 1 / value;
    return *this * inv_value;
}

Matrix operator-(const float value, Matrix& mat) {
    Matrix negative_matrix = mat * -1.0f;
    return negative_matrix + value;
}

Matrix Matrix::operator+(Matrix& other) {
    if (rows != other.rows || cols != other.cols){
        std::cerr << "Matrix dimensions must match for addition!" << std::endl;
        exit(1);
    }
    Matrix result(rows, cols);

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (cols - 1) / blockSize.x + 1,
        (rows - 1) / blockSize.y + 1
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
        (cols - 1) / blockSize.x + 1,
        (rows - 1) / blockSize.y + 1
    );

    matrix_hadamard<<<gridSize, blockSize>>>(data, other.data, result.data, rows, cols);
    cudaDeviceSynchronize();
    return result;
}

float Matrix::sum() {
    float* d_sum;
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (cols + blockSize.x - 1) / blockSize.x,
        (rows + blockSize.y - 1) / blockSize.y
    );

    matrix_sum<<<gridSize, blockSize>>>(data, d_sum, rows, cols);
    cudaDeviceSynchronize();

    float h_sum = 0.0f;
    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_sum);
    return h_sum;
}


Matrix Matrix::transpose() {
    Matrix result(cols, rows);

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (cols + blockSize.x - 1) / blockSize.x,
        (rows + blockSize.y - 1) / blockSize.y
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
        (other.cols - 1) / blockSize.x + 1,
        (rows - 1) / blockSize.y + 1
    );

    matrix_multiply<<<gridSize, blockSize>>>(data, other.data, result.data, rows, cols, other.cols);
    cudaDeviceSynchronize();
    return result;
};

Matrix Matrix::softmax() {
    int MAX_COLS = 1024;
    if (cols > MAX_COLS){
        std::cerr << "Softmax kernel doesn't support cols > " << MAX_COLS << std::endl;
        exit(1);
    }
    Matrix result(rows, cols);

    dim3 blockSize(1, MAX_COLS);
    dim3 gridSize(1, 1);

    matrix_softmax_over_rows<<<gridSize, blockSize>>>(data, result.data, rows, cols);
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

void Matrix::random(unsigned long seed, float min, float max) {
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (cols + blockSize.x - 1) / blockSize.x,
        (rows + blockSize.y - 1) / blockSize.y
    );

    fill_with_random<<<gridSize, blockSize>>>(data, seed, rows, cols, min, max);
    cudaDeviceSynchronize();
};

Matrix Matrix::get_ce_loss(Matrix& labels) {
    if (rows != labels.rows) {
        std::cerr << "Non-matching number of rows for input and labels" << std::endl;
        exit(1);
    }

    Matrix losses = Matrix(rows, 1);

    dim3 blockSize(1, 1024);
    dim3 gridSize(1, 1);

    ce_loss<<<gridSize, blockSize>>>(data, labels.data, losses.data, rows, cols);
    cudaDeviceSynchronize();
    return losses;
};

//  labels => (bsz, 1) => represents the index of the correct output
//  softmax_output => (bsz, num_classes)
Matrix ce_softmax_bwd(Matrix& labels, Matrix& softmax_output) {
    int bsz = softmax_output.getRows();
    int num_classes = softmax_output.getCols();

    if (labels.getRows() != bsz) {
        std::cerr << "Non-matching number of rows for input and labels" << std::endl;
        exit(1);
    }

    Matrix softmax_grads = Matrix(bsz, num_classes);

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (num_classes + blockSize.x - 1) / blockSize.x,
        (bsz + blockSize.y - 1) / blockSize.y
    );

    softmax_bwd<<<gridSize, blockSize>>>(labels.getDataPtr(), softmax_output.getDataPtr(), softmax_grads.getDataPtr(), bsz, num_classes);
    cudaDeviceSynchronize();
    return softmax_grads;
}
