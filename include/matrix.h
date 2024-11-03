#ifndef MATRIX_H
#define MATRIX_H

struct Matrix {
  float *data;
  int rows;
  int cols;
  int numel;

  Matrix(); 
  Matrix(int rows, int cols);

  ~Matrix();

  Matrix(const Matrix& other);

  void setData(const float* host_data);
  void getData(float* host_data) const;

  void random(const unsigned long seed, const float min, const float max);
  Matrix& operator=(const Matrix& other);
};

// Matrix-only ops
float matabsmax(const Matrix& mat);
float matsum(const Matrix& mat);
Matrix transpose(const Matrix& mat);
Matrix softmax(const Matrix& mat);
Matrix sigmoid(const Matrix& mat);
Matrix relu(const Matrix& mat);

// Matrix-Scalar ops
Matrix operator+(const Matrix& mat, const float value);
Matrix operator-(const Matrix& mat, const float value);
Matrix operator*(const Matrix& mat, const float value);
Matrix operator/(const Matrix& mat, const float value);

// Matrix-Matrix ops
Matrix operator+(const Matrix& mat1, const Matrix& mat2);
Matrix operator*(const Matrix& mat1, const Matrix& mat2);
Matrix matmul(const Matrix& mat1, const Matrix& mat2);
Matrix relu_backward(const Matrix& mat1, const Matrix& grad_output);
Matrix get_ce_loss(const Matrix& mat1, const Matrix& labels);
Matrix ce_softmax_bwd(const Matrix& labels, const Matrix& softmax_output);

#endif
