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
      Matrix(); 
      Matrix(int rows, int cols);

      ~Matrix();

      Matrix(const Matrix& other);

      void setData(const float* host_data);

      void getData(float* host_data);

      Matrix& operator=(const Matrix& other);
      
      Matrix operator+(const float value);
      Matrix operator-(const float value);
      Matrix operator*(const float value);
      Matrix operator/(const float value);

      float sum();
      
      Matrix operator+(Matrix& other); // TODO: Make const arg
      Matrix operator*(Matrix& other); // TODO: Make const arg

      Matrix transpose();

      Matrix matmul(const Matrix& other);
      Matrix softmax();
      Matrix sigmoid();
      Matrix get_ce_loss(Matrix& labels);
      void random(unsigned long seed, float min, float max);

      float* getDataPtr(){
        return data;
      }

      int getRows(){
        return rows;
      }

      int getCols(){
        return cols;
      }
      
      int getNumel(){
        return numel;
      }
};

// Non-member operator
Matrix operator-(const float value, Matrix& mat);
Matrix ce_softmax_bwd(Matrix& label, Matrix& softmax_output);

#endif
