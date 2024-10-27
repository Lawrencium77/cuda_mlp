#include "model.h"

SingleLayerPerceptron::SingleLayerPerceptron(int dim_out, int dim_in, bool sigmoid)
    : dim_out(dim_out),
      dim_in(dim_in),
      weights(dim_out, dim_in),
      grads(dim_out, dim_in),
      sigmoid(sigmoid) {}

void SingleLayerPerceptron::randomise(unsigned long seed) {
  weights.random(seed);
}

Matrix SingleLayerPerceptron::forward(Matrix& input) {
  // weights: dim_out x dim_in
  // input: dim_in x bsz
  // output: dim_out x bsz
  inputs = input; // Store for backward pass
  Matrix intermediate = weights.matmul(input); 
  return sigmoid ? intermediate.sigmoid() : intermediate;
}

Matrix SingleLayerPerceptron::backward(Matrix& grad) {
  // weights: dim_out x dim_in
  // grad: dim_out x bsz
  // output: dim_in x bsz
  if (sigmoid) {
    Matrix intermediate = weights.matmul(inputs).sigmoid();
    Matrix sigmoid_one_minus = (1.0f - intermediate); // TODO: Use M * (1.0f - M)
    Matrix sigmoid_grad = intermediate * sigmoid_one_minus;
    Matrix delta = grad * sigmoid_grad; 
    
    grads = delta.matmul(inputs.transpose());
    Matrix input_grads = weights.transpose().matmul(delta);
    return input_grads;
  } else {
    Matrix delta = grad;
    grads = delta.matmul(inputs.transpose());
    Matrix input_grads = weights.transpose().matmul(delta);
    return input_grads;
  }
}

void SingleLayerPerceptron::update_weights(float lr) {
  Matrix update = grads * (-1 * lr);
  weights = weights + update;
}



MLP::MLP(int feat_dim, int num_layers) : feat_dim(feat_dim), num_layers(num_layers) {
    for (int i = 0; i < num_layers; ++i) {
        layers.push_back(SingleLayerPerceptron(feat_dim, feat_dim));
    }
    layers.push_back(SingleLayerPerceptron(output_classes, feat_dim, false));
}

Matrix MLP::forward(Matrix& input){
    Matrix y = input;
    for (int i = 0; i < num_layers; ++i) {
        y = layers[i].forward(y);
    }
    y = layers[num_layers].forward(y); // Classification head
    return y.softmax();
}

void MLP::backward(Matrix& labels, Matrix& preds){
    Matrix grads = ce_softmax_bwd(labels, preds);
    for (int i = num_layers; i >= 0; --i) {
        grads = layers[i].backward(grads);
    }
}

void MLP::update_weights(float lr) {
  for (int i = 0; i < num_layers + 1; ++i) {
      layers[i].update_weights(lr);
  }
}

void MLP::randomise(unsigned long seed) {
    for (int i = 0; i < num_layers + 1; ++i) { // Randomise each layer, and classification head
        layers[i].randomise(seed + i); 
    }
}
