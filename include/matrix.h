// Defines Matrix class

#ifndef MATRIX_H
#define MATRIX_H

class Matrix {
  private:
      float *data;
      int rows;
      int cols;
      int numel;

  public:
      Matrix(int rows, int cols);

      ~Matrix();

      Matrix(const Matrix& other);

      void setData(const float* host_data);

      void getData(float* host_data);

      Matrix& operator=(const Matrix& other);
      
      Matrix operator+(const float value);
      Matrix operator*(const float value);
      
      Matrix operator+(Matrix& other); // TODO: Make const arg
      Matrix operator*(Matrix& other); // TODO: Make const arg

      Matrix matmul(const Matrix& other);
      Matrix softmax();
      Matrix sigmoid();
      Matrix get_ce_loss(Matrix& labels);
      void random(unsigned long seed);

      float* getDataPtr(){
        return data;
      }

      int getNumel(){
        return numel;
      }
};

#endif
