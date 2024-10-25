#include "test_utils.h"
#include "model.h"

void testSingleLayerForward(float* data, Matrix& input, int feat_dim, int bsz) {
  SingleLayerPerceptron slp(feat_dim, feat_dim);
  slp.randomise(0);
  Matrix output = slp.forward(input);
  
  output.getData(data);
  printMatrixData(data, feat_dim, bsz);
}

void testMLPForward(float* data, Matrix& input, int feat_dim, int num_layers, int bsz) {
  MLP mlp(feat_dim, num_layers);
  mlp.randomise(0);
  Matrix output = mlp.forward(input);
  
  output.getData(data);
  printMatrixData(data, feat_dim, bsz);
}

void setHostDataToConst(float* data, int numel, float value) {
  for (int i = 0; i < numel; i++){
      data[i] = value;
  }
}

void runTests() {
  int feat_dim = 8;
  int bsz = 4;
  int numel = feat_dim * bsz;
  
  float* data = new float[numel];
  setHostDataToConst(data, numel, 1.0f);

  Matrix input = Matrix(feat_dim, bsz);
  input.setData(data);
  testSingleLayerForward(data, input, feat_dim, bsz);

  int num_layers = 4;
  setHostDataToConst(data, numel, 1.0f);
  testMLPForward(data, input, feat_dim, num_layers, bsz);

  delete [] data;
}

int main() {
    runTests();
    return 0;
}
