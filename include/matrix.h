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
      
      Matrix operator+(Matrix& other); // TODO: Make const arg
      Matrix operator*(Matrix& other); // TODO: Make const arg
      Matrix matmul(const Matrix& other);
      Matrix softmax();
      Matrix sigmoid();
      void random(unsigned long seed);

      float* getDataPtr(){
        return data;
      }

      int getNumel(){
        return numel;
      }
};

#endif
