#ifndef MATRIX_H
#define MATRIX_H

#include "allocator.h"
#include "cuda_utils.h"
#include <memory>
#include <string>

struct Matrix {
  static std::unique_ptr<AllocatorBase> allocator;
  float *host_data;
  float *device_data;
  int rows;
  int cols;
  int numel;

  Matrix();
  Matrix(int rows, int cols);
  ~Matrix();

  // Use move constructor instead of copy constructor
  // Shouldn't matter too much since the compiler should use RVO
  Matrix(Matrix &&other);
  Matrix(const Matrix &other) = delete;

  Matrix &operator=(Matrix &&other);
  Matrix &operator=(const Matrix &other);

  void toDevice();
  void toHost();

  void setHostData(float *data);

  void random(const unsigned long seed, const float min, const float max);

  void printData(std::string message = "");
};

// Matrix-only ops
float matsum(const Matrix &mat);
Matrix transpose(const Matrix &mat);
Matrix softmax(const Matrix &mat);
Matrix sigmoid(const Matrix &mat);
Matrix relu(const Matrix &mat);

// Matrix-Scalar ops
Matrix operator+(const Matrix &mat, const float value);
Matrix operator-(const Matrix &mat, const float value);
Matrix operator*(const Matrix &mat, const float value);
Matrix operator/(const Matrix &mat, const float value);

// Matrix-Matrix ops
Matrix operator+(const Matrix &mat1, const Matrix &mat2);
Matrix operator*(const Matrix &mat1, const Matrix &mat2);
Matrix matmul(const Matrix &mat1, const Matrix &mat2);
Matrix relu_backward(const Matrix &mat1, const Matrix &grad_output);
Matrix get_ce_loss(const Matrix &mat1, const Matrix &labels);
Matrix ce_softmax_bwd(const Matrix &labels, const Matrix &softmax_output);
std::pair<Matrix, Matrix> get_ce_loss_and_accuracy(const Matrix &mat1,
                                                   const Matrix &labels);

#endif
