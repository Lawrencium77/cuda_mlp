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

void testAdd(float* data, int rows, int cols) {
    Matrix matrix1(rows, cols);
    Matrix matrix2(rows, cols);
    matrix1.setData(data);
    matrix2.setData(data);

    Matrix output = matrix1 + matrix2;
    
    output.getData(data);
    printData(data, rows, cols);
}

void testHadamard(float* data, int rows, int cols) {
    Matrix matrix1(rows, cols);
    Matrix matrix2(rows, cols);
    matrix1.setData(data);
    matrix2.setData(data);

    Matrix output = matrix1 * matrix2;
    
    output.getData(data);
    printData(data, rows, cols);
}

void testMul(float* data, int rows, int cols) {
    Matrix matrix1(rows, cols);
    Matrix matrix2(rows, cols);
    matrix1.setData(data);
    matrix2.setData(data);

    Matrix output = matrix1.matmul(matrix2);
    
    output.getData(data);
    printData(data, rows, cols);
}

void testSoftmax(float* data, int rows, int cols) {
    Matrix matrix1(rows, cols);
    matrix1.setData(data);

    Matrix output = matrix1.softmax();

    output.getData(data);
    printData(data, rows, cols);
}

void testSigmoid(float* data, int rows, int cols) {
    Matrix matrix1(rows, cols);
    matrix1.setData(data);

    Matrix output = matrix1.sigmoid();

    output.getData(data);
    printData(data, rows, cols);
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

    std::cout << "Testing Add Op" << std::endl;
    testAdd(data, rows, cols);

    std::cout << "Testing Hadamard Op" << std::endl;
    testHadamard(data, rows, cols);

    std::cout << "Testing Mul Op" << std::endl;
    testMul(data, rows, cols);

    std::cout << "Testing Softmax Op" << std::endl;
    testSoftmax(data, rows, cols);

    std::cout << "Testing Sigmoid Op" << std::endl;
    testSigmoid(data, rows, cols);

    delete [] data;
}

int main() {
    runTests();
    return 0;
}
