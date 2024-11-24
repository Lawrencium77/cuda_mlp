#include "model.h"
#include <iostream>

template <typename T>
void testSingleLayerForward(Matrix<T>& input, int bsz, int feat_dim) {
  SingleLayerPerceptron<T> slp(feat_dim, feat_dim);
  slp.randomise(0);
  Matrix<T>& output = slp.forward(input);
  output.printData("SLP Output");
}

template <typename T>
void testSingleLayerBackward(Matrix<T>& input, int bsz, int feat_dim) {
  SingleLayerPerceptron<T> slp(feat_dim, feat_dim);
  slp.randomise(0);
  Matrix<T>& output = slp.forward(input);
  Matrix<T> grad = slp.backward(output);
  grad.printData("SLP Gradient");
}

template <typename T>
Matrix<T> getLabels(int bsz) {
  float* labels_data = new float[bsz];
  for (int i = 0; i < bsz; i++){
      labels_data[i] = 0;
  }
  Matrix<T> labels(bsz, 1);
  labels.setHostData(labels_data);
  labels.toDevice();
  return labels;
}

template <typename T>
void testMLPForward(Matrix<T>& input, int bsz, int feat_dim, int num_layers, int num_classes) {
  MLP<T> mlp(feat_dim, num_layers);
  mlp.randomise(0);
  Matrix<T> output = mlp.forward(input);
  output.printData("MLP Output");

  Matrix<T> labels = getLabels<T>(bsz);

  Matrix<T> losses = get_ce_loss(output, labels);
  losses.printData("MLP Losses");
}

template <typename T>
void testMLPBackward(Matrix<T>& input, int feat_dim, int num_layers, int num_classes, int bsz) {
  MLP<T> mlp(feat_dim, num_layers);
  mlp.randomise(0);
  Matrix<T> output = mlp.forward(input);
  Matrix<T> labels = getLabels<T>(bsz);
  Matrix<T> losses = get_ce_loss(output, labels);

  mlp.backward(labels, output);

  float* data = new float[feat_dim * feat_dim];
  Matrix<T>& grads = mlp.layers[0].grads;
  grads.printData("First layer gradients");

  float lr = 1.0;
  mlp.layers[0].weights.printData("Before weight update");
  mlp.update_weights(lr);
  mlp.layers[0].weights.printData("After weight update");
}

void setHostDataToConst(float* data, int numel, float value) {
  for (int i = 0; i < numel; i++){
      data[i] = value;
  }
}

template <typename T>
void runTests() {
  int feat_dim = 8;
  int bsz = 4;
  int num_layers = 4;
  int num_classes = 10;
  int input_numel = feat_dim * bsz;

  float* input_data = new float[input_numel];
  setHostDataToConst(input_data, input_numel, 1.0f);
  Matrix<T> input = Matrix<T>(bsz, feat_dim);
  input.setHostData(input_data);
  input.toDevice();

  std::cout << "Testing Single Layer" << std::endl;
  testSingleLayerForward<T>(input, bsz, feat_dim);
  testSingleLayerBackward<T>(input, bsz, feat_dim);

  std::cout << "Testing MLP" << std::endl;
  setHostDataToConst(input.host_data, input_numel, 1.0f);
  testMLPForward<T>(input, bsz, feat_dim, num_layers, num_classes);
  testMLPBackward<T>(input, feat_dim, num_layers, num_classes, bsz);
}

int main() {
    runTests<float>();
    runTests<__half>();
    baseMatrix::allocator.cleanup();
    return 0;
}
