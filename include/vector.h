// Defines Vector class

#ifndef VECTOR_H
#define VECTOR_H

class Vector {
  private:
      float *data;
      int size;

  public:
      Vector(int size);

      ~Vector();

      void setData(const float* host_data);

      void getData(float* host_data);

      Vector operator+(const float value);
      Vector operator-(const float value);
      Vector operator*(const float value);
      Vector operator/(const float value);

      Vector operator+(Vector& other);
      Vector operator-(Vector& other);
      Vector operator*(Vector& other);
      Vector operator/(Vector& other);

      float* getDataPtr(){
        return data;
      }
};

#endif
