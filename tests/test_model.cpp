#include "model.h"
#include <iostream>

void testSingleLayerForward(Matrix &input, int bsz, int feat_dim) {
  SingleLayerPerceptron slp(feat_dim, feat_dim);
  slp.randomise(0);
  Matrix &output = slp.forward(input);
  output.printData("SLP Output");
}

void testSingleLayerBackward(Matrix &input, int bsz, int feat_dim) {
  SingleLayerPerceptron slp(feat_dim, feat_dim);
  slp.randomise(0);
  Matrix &output = slp.forward(input);
  Matrix grad = slp.backward(output);
  grad.printData("SLP Gradient");
}

Matrix getLabels(int bsz) {
  float *labels_data = new float[bsz];
  for (int i = 0; i < bsz; i++) {
    labels_data[i] = 0;
  }
  Matrix labels(bsz, 1);
  labels.setHostData(labels_data);
  labels.toDevice();
  return labels;
}

void testMLPForward(Matrix &input, int bsz, int feat_dim, int num_layers,
                    int num_classes) {
  MLP mlp(feat_dim, num_layers);
  mlp.randomise(0);
  Matrix output = mlp.forward(input);
  output.printData("MLP Output");

  Matrix labels = getLabels(bsz);

  Matrix losses = get_ce_loss(output, labels);
  losses.printData("MLP Losses");
}

void testMLPBackward(Matrix &input, int feat_dim, int num_layers,
                     int num_classes, int bsz) {
  MLP mlp(feat_dim, num_layers);
  mlp.randomise(0);
  Matrix output = mlp.forward(input);
  Matrix labels = getLabels(bsz);
  Matrix losses = get_ce_loss(output, labels);

  mlp.backward(labels, output);

  float *data = new float[feat_dim * feat_dim];
  Matrix &grads = mlp.layers[0].grads;
  grads.printData("First layer gradients");

  float lr = 1.0;
  mlp.layers[0].weights.printData("Before weight update");
  mlp.update_weights(lr);
  mlp.layers[0].weights.printData("After weight update");
}

void setHostDataToConst(float *data, int numel, float value) {
  for (int i = 0; i < numel; i++) {
    data[i] = value;
  }
}

void runTests() {
  int feat_dim = 8;
  int bsz = 4;
  int num_layers = 4;
  int num_classes = 10;
  int input_numel = feat_dim * bsz;
  int output_numel = num_classes * bsz;

  float *input_data = new float[input_numel];
  setHostDataToConst(input_data, input_numel, 1.0f);
  Matrix input = Matrix(bsz, feat_dim);
  input.setHostData(input_data);
  input.toDevice();

  std::cout << "Testing Single Layer" << std::endl;
  testSingleLayerForward(input, bsz, feat_dim);
  testSingleLayerBackward(input, bsz, feat_dim);

  std::cout << "Testing MLP" << std::endl;
  setHostDataToConst(input.host_data, input_numel, 1.0f);
  testMLPForward(input, bsz, feat_dim, num_layers, num_classes);
  testMLPBackward(input, feat_dim, num_layers, num_classes, bsz);
}

int main() {
  runTests();
  Matrix::allocator.cleanup();
  return 0;
}
