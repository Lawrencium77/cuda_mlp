// TODO: Add dtype conversion ops
#ifndef MATRIX_H
#define MATRIX_H

#include "allocator.h"
#include "cuda_utils.h"
#include "matrix_kernels.h"
#include <iostream>
#include <string>

// Non-templated base class. Required such that memory allocator is shared
// across Matrix objects of different dtypes.
struct baseMatrix {
  static MemoryAllocator allocator;
};

// C++ doesn't natively support FP16 dtype. So we handle all host-side data as
// float. Device-side data is handled as either FP16 or FP32.
template <typename T> struct Matrix : public baseMatrix {
  float *host_data;
  T *device_data;
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

  void random(const unsigned long seed, const T min, const T max);

  void printData(std::string message = "");
};

// Matrix-only ops
template <typename T> float matsum(const Matrix<T> &mat);

template <typename T> Matrix<T> transpose(const Matrix<T> &mat);

template <typename T> Matrix<T> softmax(const Matrix<T> &mat);

template <typename T> Matrix<T> sigmoid(const Matrix<T> &mat);

template <typename T> Matrix<T> relu(const Matrix<T> &mat);

// Matrix-Scalar ops
template <typename T>
Matrix<T> operator+(const Matrix<T> &mat, const float value);

template <typename T>
Matrix<T> operator-(const Matrix<T> &mat, const float value);

template <typename T>
Matrix<T> operator*(const Matrix<T> &mat, const float value);

template <typename T>
Matrix<T> operator/(const Matrix<T> &mat, const float value);

// Matrix-Matrix ops
template <typename T>
Matrix<T> operator+(const Matrix<T> &mat1, const Matrix<T> &mat2);

template <typename T>
Matrix<T> operator*(const Matrix<T> &mat1, const Matrix<T> &mat2);

template <typename T>
Matrix<T> matmul(const Matrix<T> &mat1, const Matrix<T> &mat2);

template <typename T>
Matrix<T> relu_backward(const Matrix<T> &mat1, const Matrix<T> &grad_output);

template <typename T>
Matrix<T> get_ce_loss(const Matrix<T> &mat1, const Matrix<T> &labels);

template <typename T>
Matrix<T> ce_softmax_bwd(const Matrix<T> &labels,
                         const Matrix<T> &softmax_output);

template <typename T>
std::pair<Matrix<T>, Matrix<T>>
get_ce_loss_and_accuracy(const Matrix<T> &mat1, const Matrix<T> &labels);

#include "matrix.cuh"

#endif
