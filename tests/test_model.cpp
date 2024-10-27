#include "test_utils.h"
#include "model.h"

void testSingleLayerForward(float* data, Matrix& input, int feat_dim, int bsz) {
  SingleLayerPerceptron slp(feat_dim, feat_dim);
  slp.randomise(0);
  Matrix output = slp.forward(input);
  
  output.getData(data);
  printMatrixData(data, feat_dim, bsz);
}

void testSingleLayerBackward(float* data, Matrix& input, int feat_dim, int bsz) {
  SingleLayerPerceptron slp(feat_dim, feat_dim);
  slp.randomise(0);
  Matrix output = slp.forward(input);
  Matrix grad = slp.backward(input);
  
  grad.getData(data);
  printMatrixData(data, feat_dim, bsz);
}

void testMLPForward(float* data, Matrix& input, int feat_dim, int num_layers, int num_classes, int bsz) {
  MLP mlp(feat_dim, num_layers);
  mlp.randomise(0);
  Matrix output = mlp.forward(input);
  
  output.getData(data);
  printMatrixData(data, num_classes, bsz);

  float* labels_data = new float[bsz];
  for (int i = 0; i < bsz; i++){
      labels_data[i] = 0;
  }
  Matrix labels(1, bsz);
  labels.setData(labels_data);

  Matrix losses = output.get_ce_loss(labels);
  losses.getData(data);
  printMatrixData(data, 1, bsz);
}

void setHostDataToConst(float* data, int numel, float value) {
  for (int i = 0; i < numel; i++){
      data[i] = value;
  }
}

void runTests() {
  int feat_dim = 8;
  int bsz = 4;
  int num_classes = 10;
  int input_numel = feat_dim * bsz;
  int output_numel = num_classes * bsz;
  
  float* input_data = new float[input_numel];
  setHostDataToConst(input_data, input_numel, 1.0f);

  Matrix input = Matrix(feat_dim, bsz);
  input.setData(input_data);
  testSingleLayerForward(input_data, input, feat_dim, bsz);
  testSingleLayerBackward(input_data, input, feat_dim, bsz);

  int num_layers = 4;
  float* output_data = new float[output_numel];
  setHostDataToConst(input_data, input_numel, 1.0f);
  testMLPForward(output_data, input, feat_dim, num_layers, num_classes, bsz);

  delete [] input_data;
  delete [] output_data;
}

int main() {
    runTests();
    return 0;
}
