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

Matrix testSoftmax(Matrix& input, float* data, int rows, int cols) {
    Matrix output = input.softmax();

    output.getData(data);
    printMatrixData(data, rows, cols);
    return output;
}

void testSigmoid(float* data, int rows, int cols) {
    Matrix matrix1(rows, cols);
    matrix1.setData(data);

    Matrix output = matrix1.sigmoid();

    output.getData(data);
    printMatrixData(data, rows, cols);
}

Matrix testRandom(int rows, int cols) {
    Matrix matrix(rows, cols);
    matrix.random(0);

    float* data = new float[rows * cols];
    matrix.getData(data);
    printMatrixData(data, rows, cols);

    return matrix;
}

void testCrossEntropy(Matrix& input, int num_classes, int bsz) {
    Matrix labels(1, bsz);

    float* labels_data = new float[bsz];
    for (int i = 0; i < bsz; i++){
        labels_data[i] = 0;
    }
    labels.setData(labels_data);

    Matrix losses = input.get_ce_loss(labels);

    float* data = new float[num_classes * bsz];
    losses.getData(data);
    printMatrixData(data, 1, bsz);
}

void runTests() {
    int rows = 8;
    int cols = 8;
    int numel = rows * cols;

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

    std::cout << "Testing Sigmoid Op" << std::endl;
    testSigmoid(data, rows, cols);

    std::cout << "Testing Random Init Op" << std::endl;
    Matrix random_values = testRandom(rows, cols);

    std::cout << "Testing Softmax Op" << std::endl;
    Matrix normalised_values = testSoftmax(random_values, data, rows, cols);

    std::cout << "Testing Cross Entropy Op" << std::endl;
    testCrossEntropy(normalised_values, rows, cols);

    delete [] data;
}

int main() {
    runTests();
    return 0;
}
