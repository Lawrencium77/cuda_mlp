#include <iostream>
#include "vector.h"

void printData(const float* data, int size) {
    std::cout << "Data: ";
    for (int i = 0; i < size; i++){
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
};

int main() {
    int size = 8;

    // Create an array of floats
    float* data = new float[size];
    for (int i = 0; i < size; i++){
        data[i] = 1.0f;
    }

    Vector my_vector(size);
    my_vector.setData(data);

    float value = 10.0;
    Vector new_vector = my_vector + value;

    new_vector.getData(data);

    printData(data, size);

    delete[] data;
}
