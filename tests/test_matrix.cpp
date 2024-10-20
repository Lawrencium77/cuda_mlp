#include <iostream>
#include "matrix.h"

void printData(const float* data, int rows, int cols) {
    std::cout << "Data: \n";
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            std::cout << data[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
};

void testAdd(float* data, int size) {
}

void testMul(float* data, int size) {
}

void runTests() {
    int rows = 8;
    int cols = 8;
    int numel = rows * cols;
    float value = 10.0;

    float* data = new float[numel];
    for (int i = 0; i < numel; i++){
        data[i] = 1.0f;
    }
    
    std::cout << "Testing Print Op" << std::endl;
    printData(data, rows, cols);
}

int main() {
    runTests();
    return 0;
}
