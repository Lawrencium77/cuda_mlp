#include "matrix.h"
#include "test_utils.h"

void testSum(float* data, int bsz, int feats) {
    Matrix matrix(bsz, feats);
    matrix.setData(data);

    float sum = matrix.sum();
    std::cout << "Sum: " << sum  << "\n" << std::endl;
}

void testAddConst(float* data, float value, int bsz, int feats) {
    Matrix matrix(bsz, feats);
    matrix.setData(data);

    Matrix output = matrix + value;
    
    output.getData(data);
    printMatrixData(data, bsz, feats);
}

void testMulConst(float* data, float value, int bsz, int feats) {
    Matrix matrix(bsz, feats);
    matrix.setData(data);

    Matrix output = matrix * value;
    
    output.getData(data);
    printMatrixData(data, bsz, feats);
}

void testAdd(float* data, int bsz, int feats) {
    Matrix matrix1(bsz, feats);
    Matrix matrix2(bsz, feats);
    matrix1.setData(data);
    matrix2.setData(data);

    Matrix output = matrix1 + matrix2;
    
    output.getData(data);
    printMatrixData(data, bsz, feats);
}

void testHadamard(float* data, int bsz, int feats) {
    Matrix matrix1(bsz, feats);
    Matrix matrix2(bsz, feats);
    matrix1.setData(data);
    matrix2.setData(data);

    Matrix output = matrix1 * matrix2;
    
    output.getData(data);
    printMatrixData(data, bsz, feats);
}

void testTranspose(Matrix& input, float* data, int bsz, int feats) {
    Matrix output = input.transpose();
    output.getData(data);
    printMatrixData(data, feats, bsz);
}

void testMul(float* data, int bsz, int feats) {
    Matrix matrix1(bsz, feats);
    Matrix matrix2(bsz, feats);
    matrix1.setData(data);
    matrix2.setData(data);

    Matrix output = matrix1.matmul(matrix2);
    
    output.getData(data);
    printMatrixData(data, bsz, feats);
}

Matrix testSoftmax(Matrix& input, float* data, int bsz, int feats) {
    Matrix output = input.softmax();

    output.getData(data);
    printMatrixData(data, bsz, feats);
    return output;
}

void testSigmoid(float* data, int bsz, int feats) {
    Matrix matrix1(bsz, feats);
    matrix1.setData(data);

    Matrix output = matrix1.sigmoid();

    output.getData(data);
    printMatrixData(data, bsz, feats);
}

Matrix testRandom(int bsz, int feats) {
    Matrix matrix(bsz, feats);
    matrix.random(0, -1.0f, 1.0f);

    float* data = new float[bsz * feats];
    matrix.getData(data);
    printMatrixData(data, bsz, feats);

    return matrix;
}


void testCrossEntropy(Matrix& input, int num_classes, int bsz) {
    Matrix labels(bsz, 1);

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

void testCESoftmaxBwd(int bsz, int feats) {
    Matrix labels(1, feats);
    Matrix softmax_out(bsz, feats);

    softmax_out.random(0, -1.0f, 1.0f);
    float* labels_data = new float[feats];
    for (int i = 0; i < feats; i++){
        labels_data[i] = 0;
    }

    Matrix output = ce_softmax_bwd(labels, softmax_out);
    
    float* data = new float[bsz * feats];
    output.getData(data);
    printMatrixData(data, bsz, feats);
}

void runTests() {
    int bsz = 8;
    int feats = 8;
    int numel = bsz * feats;
    float value = 1.0f;

    float* data = new float[numel];
    for (int i = 0; i < numel; i++){
        data[i] = 1.0f;
    }
    
    std::cout << "Testing Print Op" << std::endl;
    printMatrixData(data, bsz, feats);

    std::cout << "Testing Sum Op" << std::endl;
    testSum(data, bsz, feats);

    std::cout << "Testing Add Const Op" << std::endl;
    testAddConst(data, value, bsz, feats);

    std::cout << "Testing Mul Const Op" << std::endl;
    testMulConst(data, value, bsz, feats);

    std::cout << "Testing Add Op" << std::endl;
    testAdd(data, bsz, feats);

    std::cout << "Testing Hadamard Op" << std::endl;
    testHadamard(data, bsz, feats);

    std::cout << "Testing Mul Op" << std::endl;
    testMul(data, bsz, feats);

    std::cout << "Testing Sigmoid Op" << std::endl;
    testSigmoid(data, bsz, feats);

    std::cout << "Testing Random Init Op" << std::endl;
    Matrix random_values = testRandom(bsz, feats);

    std::cout << "Testing Transpose Op" << std::endl;
    testTranspose(random_values, data, bsz, feats);

    std::cout << "Testing Softmax Op" << std::endl;
    Matrix normalised_values = testSoftmax(random_values, data, bsz, feats);

    std::cout << "Testing Cross Entropy Op" << std::endl;
    testCrossEntropy(normalised_values, feats, bsz);

    std::cout << "Testing CE + Softmax Bwd" << std::endl;
    testCESoftmaxBwd(bsz, feats);

    delete [] data;
}

int main() {
    runTests();
    return 0;
}
