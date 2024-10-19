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
