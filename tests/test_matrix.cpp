#include "matrix.h"
#include "test_utils.h"

void testAdd(float* data, int rows, int cols) {
    Matrix matrix1(rows, cols);
    Matrix matrix2(rows, cols);
    matrix1.setData(data);
    matrix2.setData(data);

    Matrix output = matrix1 + matrix2;
    
    output.getData(data);
    printMatrixData(data, rows, cols);
}

void testHadamard(float* data, int rows, int cols) {
    Matrix matrix1(rows, cols);
    Matrix matrix2(rows, cols);
    matrix1.setData(data);
    matrix2.setData(data);

    Matrix output = matrix1 * matrix2;
    
    output.getData(data);
    printMatrixData(data, rows, cols);
}

void testMul(float* data, int rows, int cols) {
    Matrix matrix1(rows, cols);
    Matrix matrix2(rows, cols);
    matrix1.setData(data);
    matrix2.setData(data);

    Matrix output = matrix1.matmul(matrix2);
    
    output.getData(data);
    printMatrixData(data, rows, cols);
}

void testSoftmax(float* data, int rows, int cols) {
    Matrix matrix1(rows, cols);
    matrix1.setData(data);

    Matrix output = matrix1.softmax();

    output.getData(data);
    printMatrixData(data, rows, cols);
}

void testSigmoid(float* data, int rows, int cols) {
    Matrix matrix1(rows, cols);
    matrix1.setData(data);

    Matrix output = matrix1.sigmoid();

    output.getData(data);
    printMatrixData(data, rows, cols);
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
    printMatrixData(data, rows, cols);

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
