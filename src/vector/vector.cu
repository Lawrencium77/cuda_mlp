// Implements Vector class

#include "vector.h"
#include "vector_kernels.h"
#include <iostream>

Vector::Vector(int size) : size(size) {
    cudaMalloc(&data, size * sizeof(float));
}

Vector::~Vector() {
    cudaFree(data);
}

void Vector::setData(const float* host_data) {
    cudaMemcpy(data, host_data, size * sizeof(float), cudaMemcpyHostToDevice);
}

void Vector::getData(float* host_data) {
    cudaMemcpy(host_data, data, size * sizeof(float), cudaMemcpyDeviceToHost);
}

Vector Vector::operator+(const float value) {
    Vector result(size);
    vector_add_const<<<1, size>>>(data, value, result.data, size);
    cudaDeviceSynchronize();
    return result;
}

Vector Vector::operator-(const float value) {
    Vector result(size);
    vector_sub_const<<<1, size>>>(data, value, result.data, size);
    cudaDeviceSynchronize();
    return result;
}

Vector Vector::operator*(const float value) {
    Vector result(size);
    vector_mul_const<<<1, size>>>(data, value, result.data, size);
    cudaDeviceSynchronize();
    return result;
}

Vector Vector::operator/(const float value) {
    Vector result(size);
    vector_div_const<<<1, size>>>(data, value, result.data, size);
    cudaDeviceSynchronize();
    return result;
}

Vector Vector::operator+(Vector& other) {
    Vector result(size);
    float* other_data = other.getDataPtr();
    vector_add<<<1, size>>>(data, other_data, result.data, size);
    cudaDeviceSynchronize();
    return result;
}

Vector Vector::operator-(Vector& other) {
    Vector result(size);
    float* other_data = other.getDataPtr();
    vector_sub<<<1, size>>>(data, other_data, result.data, size);
    cudaDeviceSynchronize();
    return result;
}

Vector Vector::operator*(Vector& other) {
    Vector result(size);
    float* other_data = other.getDataPtr();
    vector_mul<<<1, size>>>(data, other_data, result.data, size);
    cudaDeviceSynchronize();
    return result;
}

Vector Vector::operator/(Vector& other) {
    Vector result(size);
    float* other_data = other.getDataPtr();
    vector_div<<<1, size>>>(data, other_data, result.data, size);
    cudaDeviceSynchronize();
    return result;
}

Vector Vector::sigmoid(){
    Vector result(size);
    vector_sigmoid<<<1, size>>>(data, result.data, size);
    cudaDeviceSynchronize();
    return result;
}
