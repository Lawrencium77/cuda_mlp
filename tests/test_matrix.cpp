#include "matrix.h"
#include "test_utils.h"

void testPrintOp(Matrix& input1, Matrix& input2, float* data, int bsz, int feats) {
    input1.getData(data);
    printMatrixData(data, bsz, feats, "Input 1");

    input2.getData(data);
    printMatrixData(data, bsz, feats, "Input 2");
}

void testSum(Matrix& input1, float* data, int bsz, int feats) {
    float sum = matsum(input1);
    std::cout << "Sum for input1: " << sum  << "\n" << std::endl;
}

void testTranspose(Matrix& input1, float* data, int bsz, int feats) {
    Matrix output = transpose(input1);
    output.getData(data);
    printMatrixData(data, feats, bsz);
}

void testAddConst(Matrix& input1, float* data, float value, int bsz, int feats) {
    Matrix output = input1 + value;
    output.getData(data);
    printMatrixData(data, bsz, feats);
}

void testMulConst(Matrix& input1, float* data, float value, int bsz, int feats) {
    Matrix output = input1 * value;
    output.getData(data);
    printMatrixData(data, bsz, feats);
}

void testAdd(Matrix& input1, Matrix& input2, float* data, int bsz, int feats) {
    Matrix output = input1 + input2;
    output.getData(data);
    printMatrixData(data, bsz, feats);
}

void testHadamard(Matrix& input1, Matrix& input2, float* data, int bsz, int feats) {
    Matrix output = input1 * input2;
    output.getData(data);
    printMatrixData(data, bsz, feats);
}

void testMul(Matrix& input1, Matrix& input2, float* data, int bsz) {
    Matrix output = matmul(input1, transpose(input2));
    output.getData(data);
    printMatrixData(data, bsz, bsz);
}

Matrix testSoftmax(Matrix& input, float* data, int bsz, int feats) {
    Matrix output = softmax(input);

    output.getData(data);
    printMatrixData(data, bsz, feats);
    return output;
}

void testSigmoid(Matrix& input1, float* data, int bsz, int feats) {
    Matrix output = sigmoid(input1);
    output.getData(data);
    printMatrixData(data, bsz, feats);
}

void testRelu(Matrix& input1, float* data, int bsz, int feats) {
    Matrix output = relu(input1);
    output.getData(data);
    printMatrixData(data, bsz, feats);
}

void testCrossEntropy(Matrix& input, int num_classes, int bsz) {
    Matrix labels(bsz, 1);

    float* labels_data = new float[bsz];
    for (int i = 0; i < bsz; i++){
        labels_data[i] = 0;
    }
    labels.setData(labels_data);

    Matrix losses = get_ce_loss(input, labels);

    float* data = new float[num_classes * bsz];
    losses.getData(data);
    printMatrixData(data, 1, bsz);
}

void testCESoftmaxBwd(Matrix& softmax_out, int bsz, int feats) {
    Matrix labels(bsz, 1);

    float* labels_data = new float[bsz];
    for (int i = 0; i < bsz; i++){
        labels_data[i] = 0;
    }
    labels.setData(labels_data);

    Matrix output = ce_softmax_bwd(labels, softmax_out);
    
    float* data = new float[bsz * feats];
    output.getData(data);
    printMatrixData(data, bsz, feats);
}

void runTests() {
    int bsz = 2;
    int feats = 4;
    float value = 2.0f;

    Matrix input1 = Matrix(bsz, feats);
    Matrix input2 = Matrix(bsz, feats);
    input1.random(0, -1.0f, 1.0f);
    input2.random(1, -1.0f, 1.0f);
    
    int largest_dim = bsz > feats ? bsz : feats;
    float* data = new float[largest_dim * largest_dim];

    std::cout << "Testing Print" << std::endl;
    testPrintOp(input1, input2, data, bsz, feats);

    std::cout << "Testing Sum" << std::endl;
    testSum(input1, data, bsz, feats);

    std::cout << "Testing Transpose" << std::endl;
    testTranspose(input1, data, bsz, feats);

    std::cout << "Testing Add Const" << std::endl;
    testAddConst(input1, data, value, bsz, feats);

    std::cout << "Testing Mul Const" << std::endl;
    testMulConst(input1, data, value, bsz, feats);

    std::cout << "Testing Element-Wise Add" << std::endl;
    testAdd(input1, input2, data, bsz, feats);

    std::cout << "Testing Hadamard" << std::endl;
    testHadamard(input1, input2, data, bsz, feats);

    std::cout << "Testing Mul" << std::endl;
    testMul(input1, input2, data, bsz);

    std::cout << "Testing Sigmoid" << std::endl;
    testSigmoid(input1, data, bsz, feats);

    std::cout << "Testing ReLU" << std::endl;
    testRelu(input1, data, bsz, feats);

    std::cout << "Testing Softmax" << std::endl;
    Matrix normalised_values = testSoftmax(input1, data, bsz, feats);

    std::cout << "Testing Cross Entropy" << std::endl;
    testCrossEntropy(normalised_values, feats, bsz);

    std::cout << "Testing CE + Softmax Bwd" << std::endl;
    testCESoftmaxBwd(normalised_values, bsz, feats);

    delete [] data;
}

int main() {
    runTests();
    return 0;
}
