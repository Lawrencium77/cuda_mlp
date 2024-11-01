#include "model.h"
#include <math.h>

SingleLayerPerceptron::SingleLayerPerceptron(int dim_out, int dim_in, bool sigmoid)
    : dim_out(dim_out),
      dim_in(dim_in),
      weights(dim_in, dim_out),
      grads(dim_in, dim_out),
      sigmoid(sigmoid) {}

void SingleLayerPerceptron::randomise(unsigned long seed) {
  float max = 1.0f / sqrt(dim_in); // Xavier initialisation
  float min = -max;
  weights.random(seed, min, max);
}

Matrix SingleLayerPerceptron::forward(Matrix& input) {
  // weights: dim_in x dim_out
  // input: bsz x dim_in
  // output: bsz x dim_out
  inputs = input; // Store for backward pass
  Matrix Z = input.matmul(weights);
  activations = sigmoid ? Z.sigmoid() : Z;
  return activations;
}

Matrix SingleLayerPerceptron::backward(Matrix& grad) {
  // weights: dim_in x dim_out
  // grad: bsz x dim_out
  // input_grads: bsz x dim_in
  // weight_grads: dim_in x dim_out
  Matrix inputs_tranpose = inputs.transpose(); 
  Matrix weights_tranpose = weights.transpose(); 
  Matrix delta;
  
  if (sigmoid) {
    Matrix sigmoid_one_minus = (1.0f - activations); 
    Matrix sigmoid_grad = activations * sigmoid_one_minus; 
    delta = grad * sigmoid_grad; // (bsz x dim_out) 
  } else {
    delta = grad;
  }
  
  grads = inputs_tranpose.matmul(delta); 
  Matrix input_grads = delta.matmul(weights_tranpose); 
  return input_grads;
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
