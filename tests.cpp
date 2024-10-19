#include <iostream>
#include "vector.h"

void printData(const float* data, int size) {
    std::cout << "Data: ";
    for (int i = 0; i < size; i++){
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
};

void testAddConst(int size) {
    
    // Create input data
    float* data = new float[size];
    for (int i = 0; i < size; i++){
        data[i] = 1.0f;
    }

    // Move to GPU
    Vector my_vector(size);
    my_vector.setData(data);

    // Do addition
    float value = 10.0;
    Vector new_vector = my_vector + value;

    // Return output to CPU and print
    new_vector.getData(data);
    printData(data, size);

    delete[] data;
}

void runTests() {
    int size = 5;
    testAddConst(size);
}

int main() {
    runTests();
    return 0;
}
