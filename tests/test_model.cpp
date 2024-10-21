#include "test_utils.h"
#include "model.h"

void testSingleLayerForward(float* data, Matrix& input, int feat_dim) {
  SingleLayerPerceptron slp(feat_dim);
  Matrix output = slp.forward(input);
  
  output.getData(data);
  printMatrixData(data, feat_dim, feat_dim);
}

void runTests() {
  int feat_dim = 8;
  int numel = feat_dim * feat_dim;
  
  float* data = new float[numel];
  for (int i = 0; i < numel; i++){
      data[i] = 1.0f;
  }

  Matrix input = Matrix(feat_dim, feat_dim);
  input.setData(data);
  testSingleLayerForward(data, input, feat_dim);
}

int main() {
    runTests();
    return 0;
}
