#include "test_utils.h"
#include "model.h"

void testSingleLayerForward(float* data, Matrix& input, int bsz, int feat_dim) {
  SingleLayerPerceptron slp(feat_dim, feat_dim);
  slp.randomise(0);
  Matrix output = slp.forward(input);
  
  output.getData(data);
  printMatrixData(data, bsz, feat_dim);
}

void testSingleLayerBackward(float* data, Matrix& input, int bsz, int feat_dim) {
  SingleLayerPerceptron slp(feat_dim, feat_dim);
  slp.randomise(0);
  Matrix output = slp.forward(input);
  Matrix grad = slp.backward(output);
  
  grad.getData(data);
  printMatrixData(data, bsz, feat_dim);
}

Matrix getLabels(int bsz) {
  float* labels_data = new float[bsz];
  for (int i = 0; i < bsz; i++){
      labels_data[i] = 0;
  }
  Matrix labels(bsz, 1);
  labels.setData(labels_data);
  return labels;
}

void testMLPForward(float* data, Matrix& input, int bsz, int feat_dim, int num_layers, int num_classes) {
  MLP mlp(feat_dim, num_layers);
  mlp.randomise(0);
  Matrix output = mlp.forward(input);
  
  output.getData(data);
  printMatrixData(data, bsz, num_classes);

  Matrix labels = getLabels(bsz);

  Matrix losses = output.get_ce_loss(labels);
  losses.getData(data);
  printMatrixData(data, 1, bsz);
}

void testMLPBackward(Matrix& input, int feat_dim, int num_layers, int num_classes, int bsz) {
  MLP mlp(feat_dim, num_layers);
  mlp.randomise(0);
  Matrix output = mlp.forward(input);
  Matrix labels = getLabels(bsz);
  Matrix losses = output.get_ce_loss(labels);

  mlp.backward(labels, output);

  // Examine first layer gradient
  float* data = new float[feat_dim * feat_dim];
  Matrix grads = mlp.layers[0].grads;
  grads.getData(data);
  printMatrixData(data, feat_dim, feat_dim, "for first layer gradients");

  // Update weights
  float lr = 1.0;
  mlp.layers[0].weights.getData(data);
  printMatrixData(data, feat_dim, feat_dim, "before weight update");
  mlp.update_weights(lr);
  mlp.layers[0].weights.getData(data);
  printMatrixData(data, feat_dim, feat_dim, "after weight update");
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

  std::cout << "Testing Single Layer" << std::endl;
  Matrix input = Matrix(bsz, feat_dim);
  input.setData(input_data);
  testSingleLayerForward(input_data, input, bsz, feat_dim);
  testSingleLayerBackward(input_data, input, bsz, feat_dim);

  std::cout << "Testing MLP" << std::endl;
  int num_layers = 4;
  float* output_data = new float[output_numel];
  setHostDataToConst(input_data, input_numel, 1.0f);
  testMLPForward(output_data, input, bsz, feat_dim, num_layers, num_classes);

  // testMLPBackward(input, feat_dim, num_layers, num_classes, bsz);

  delete [] input_data;
  delete [] output_data;
}

int main() {
    runTests();
    return 0;
}
