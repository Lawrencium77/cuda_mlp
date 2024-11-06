#include "matrix.h"
#include <iostream>

void testPrintOp(Matrix& input1, Matrix& input2, float* data, int bsz, int feats) {
    input1.printData("Input 1");
    input2.printData("Input 2");
}

void testSum(Matrix& input1, float* data, int bsz, int feats) {
    float sum = matsum(input1);
    std::cout << "Sum for input1: " << sum  << "\n" << std::endl;
}

void testTranspose(Matrix& input1, float* data, int bsz, int feats) {
    Matrix output = transpose(input1);
    output.printData("Testing Transpose");
}

void testAddConst(Matrix& input1, float* data, float value, int bsz, int feats) {
    Matrix output = input1 + value;
    output.printData("Testing Add Const");
}

void testMulConst(Matrix& input1, float* data, float value, int bsz, int feats) {
    Matrix output = input1 * value;
    output.printData("Testing Mul Const");
}

void testAdd(Matrix& input1, Matrix& input2, float* data, int bsz, int feats) {
    Matrix output = input1 + input2;
    output.printData("Testing Element-Wise Add");
}

void testHadamard(Matrix& input1, Matrix& input2, float* data, int bsz, int feats) {
    Matrix output = input1 * input2;
    output.printData("Testing Hadamard");
}

void testMul(Matrix& input1, Matrix& input2, float* data, int bsz) {
    Matrix output = matmul(input1, transpose(input2));
    output.printData("Testing Mul");
}

Matrix testSoftmax(Matrix& input, float* data, int bsz, int feats) {
    Matrix output = softmax(input);
    output.printData("Testing Softmax");
    return output;
}

void testSigmoid(Matrix& input1, float* data, int bsz, int feats) {
    Matrix output = sigmoid(input1);
    output.printData("Testing Sigmoid");
}

void testRelu(Matrix& input1, float* data, int bsz, int feats) {
    Matrix output = relu(input1);
    output.printData("Testing ReLU");
}

void testCrossEntropy(Matrix& input, int num_classes, int bsz) {
    Matrix labels(bsz, 1);

    float* labels_data = new float[bsz];
    for (int i = 0; i < bsz; i++){
        labels_data[i] = 0;
    }
    labels.setHostData(labels_data);
    labels.toDevice();

    Matrix losses = get_ce_loss(input, labels);
    losses.printData("Cross Entropy Losses");
}

void testCESoftmaxBwd(Matrix& softmax_out, int bsz, int feats) {
    Matrix labels(bsz, 1);

    float* labels_data = new float[bsz];
    for (int i = 0; i < bsz; i++){
        labels_data[i] = 0;
    }
    labels.setHostData(labels_data);
    labels.toDevice();

    Matrix output = ce_softmax_bwd(labels, softmax_out);
    output.printData("CE Softmax Bwd");
}

void testAccuracy(Matrix& input, int num_classes, int bsz) {
    Matrix labels(bsz, 1);

    float* labels_data = new float[bsz];
    for (int i = 0; i < bsz; i++){
        labels_data[i] = 0;
    }
    labels.setHostData(labels_data);
    labels.toDevice();

    std::pair<Matrix, Matrix> loss_and_preds = get_ce_loss_and_accuracy(input, labels);
    float accuracy = 100 * matsum(loss_and_preds.second) / bsz;
    std::cout << "Accuracy: " << accuracy << "%\n" << std::endl;
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

    testPrintOp(input1, input2, data, bsz, feats);
    testSum(input1, data, bsz, feats);
    testTranspose(input1, data, bsz, feats);
    testAddConst(input1, data, value, bsz, feats);
    testMulConst(input1, data, value, bsz, feats);
    testAdd(input1, input2, data, bsz, feats);
    testHadamard(input1, input2, data, bsz, feats);
    testMul(input1, input2, data, bsz);
    testSigmoid(input1, data, bsz, feats);
    testRelu(input1, data, bsz, feats);
    
    Matrix normalised_values = testSoftmax(input1, data, bsz, feats);
    testCrossEntropy(normalised_values, feats, bsz);
    testCESoftmaxBwd(normalised_values, bsz, feats);
    testAccuracy(normalised_values, feats, bsz);

    delete [] data;
}

int main() {
    runTests();
    return 0;
}
