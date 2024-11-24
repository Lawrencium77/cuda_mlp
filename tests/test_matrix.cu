#include <cuda_fp16.h>
#include "matrix.h"
#include <iostream>

template <typename T>
void testPrintOp(Matrix<T>& input1, Matrix<T>& input2) {
    input1.printData("Input 1");
    input2.printData("Input 2");
}

template <typename T>
void testSum(Matrix<T>& input1, float* data, int bsz, int feats) {
    float sum = matsum(input1);
    std::cout << "Sum for input1: " << sum  << "\n" << std::endl;
}

template <typename T>
void testTranspose(Matrix<T>& input1, float* data, int bsz, int feats) {
    Matrix<T> output = transpose(input1);
    output.printData("Testing Transpose");
}

template <typename T>
void testAddConst(Matrix<T>& input1, float* data, float value, int bsz, int feats) {
    Matrix<T> output = input1 + value;
    output.printData("Testing Add Const");
}

template <typename T>
void testMulConst(Matrix<T>& input1, float* data, float value, int bsz, int feats) {
    Matrix<T> output = input1 * value;
    output.printData("Testing Mul Const");
}

template <typename T>
void testAdd(Matrix<T>& input1, Matrix<T>& input2, float* data, int bsz, int feats) {
    Matrix<T> output = input1 + input2;
    output.printData("Testing Element-Wise Add");
}

template <typename T>
void testHadamard(Matrix<T>& input1, Matrix<T>& input2, float* data, int bsz, int feats) {
    Matrix<T> output = input1 * input2;
    output.printData("Testing Hadamard");
}

template <typename T>
void testMul(Matrix<T>& input1, Matrix<T>& input2, float* data, int bsz) {
    Matrix<T> output = matmul(input1, transpose(input2));
    output.printData("Testing Mul");
}

template <typename T>
Matrix<T> testSoftmax(Matrix<T>& input, float* data, int bsz, int feats) {
    Matrix<T> output = softmax(input);
    output.printData("Testing Softmax");
    return output;
}

template <typename T>
void testSigmoid(Matrix<T>& input1, float* data, int bsz, int feats) {
    Matrix<T> output = sigmoid(input1);
    output.printData("Testing Sigmoid");
}

template <typename T>
void testRelu(Matrix<T>& input1, float* data, int bsz, int feats) {
    Matrix<T> output = relu(input1);
    output.printData("Testing ReLU");
}

template <typename T>
void testCrossEntropy(Matrix<T>& input, int num_classes, int bsz) {
    Matrix<T> labels(bsz, 1);

    float* labels_data = new float[bsz];
    for (int i = 0; i < bsz; i++){
        labels_data[i] = 0;
    }
    labels.setHostData(labels_data);
    labels.toDevice();

    Matrix<T> losses = get_ce_loss(input, labels);
    losses.printData("Cross Entropy Losses");
}

template <typename T>
void testCESoftmaxBwd(Matrix<T>& softmax_out, int bsz, int feats) {
    Matrix<T> labels(bsz, 1);

    float* labels_data = new float[bsz];
    for (int i = 0; i < bsz; i++){
        labels_data[i] = 0;
    }
    labels.setHostData(labels_data);
    labels.toDevice();

    Matrix<T> output = ce_softmax_bwd(labels, softmax_out);
    output.printData("CE Softmax Bwd");
}

template <typename T>
void testAccuracy(Matrix<T>& input, int num_classes, int bsz) {
    Matrix<T> labels(bsz, 1);

    float* labels_data = new float[bsz];
    for (int i = 0; i < bsz; i++){
        labels_data[i] = 0;
    }
    labels.setHostData(labels_data);
    labels.toDevice();

    std::pair<Matrix<T>, Matrix<T>> loss_and_preds = get_ce_loss_and_accuracy(input, labels);
    float accuracy = 100 * matsum(loss_and_preds.second) / bsz;
    std::cout << "Accuracy: " << accuracy << "%\n" << std::endl;
}

template <typename T>
void runTests() {
    int bsz = 2;
    int feats = 4;
    float value = 2.0f;

    Matrix<T> input1 = Matrix<T>(bsz, feats);
    Matrix<T> input2 = Matrix<T>(bsz, feats);
    input1.random(0, T(-1), T(1));
    input2.random(1, T(-1), T(1));

    int largest_dim = bsz > feats ? bsz : feats;
    float* data = new float[largest_dim * largest_dim];

    testPrintOp(input1, input2);
    testSum(input1, data, bsz, feats);
    testTranspose(input1, data, bsz, feats);
    testAddConst(input1, data, value, bsz, feats);
    testMulConst(input1, data, value, bsz, feats);
    testAdd(input1, input2, data, bsz, feats);
    testHadamard(input1, input2, data, bsz, feats);
    testMul(input1, input2, data, bsz);
    testSigmoid(input1, data, bsz, feats);
    testRelu(input1, data, bsz, feats);

    Matrix<T> normalised_values = testSoftmax(input1, data, bsz, feats);
    testCrossEntropy(normalised_values, feats, bsz);
    testCESoftmaxBwd(normalised_values, bsz, feats);
    testAccuracy(normalised_values, feats, bsz);

    delete [] data;
}

int main() {
    runTests<__half>();
    baseMatrix::allocator.cleanup();
    return 0;
}
