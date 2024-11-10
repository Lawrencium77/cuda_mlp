#include "model.h"
#include <math.h>

SingleLayerPerceptron::SingleLayerPerceptron(const int dim_out, const int dim_in, const bool use_activation)
    : dim_out(dim_out),
      dim_in(dim_in),
      weights(dim_in, dim_out),
      grads(dim_in, dim_out),
      use_activation(use_activation) {}

void SingleLayerPerceptron::randomise(const unsigned long seed) {
  float max = 1.0f / sqrt(dim_in); // Xavier initialisation
  float min = -max;
  weights.random(seed, min, max);
}

Matrix& SingleLayerPerceptron::forward(Matrix& input) {
  // weights: dim_in x dim_out
  // input: bsz x dim_in
  // output: bsz x dim_out
  inputs = input; // Invokes a copy
  Matrix Z = matmul(inputs, weights);
  if (use_activation) {
    activations = relu(Z);
  } else {
    activations = std::move(Z);
  }
  return activations;
}

Matrix SingleLayerPerceptron::backward(Matrix& grad) {
  // weights: dim_in x dim_out
  // grad: bsz x dim_out
  // input_grads: bsz x dim_in
  // weight_grads: dim_in x dim_out
  const int bsz = grad.rows;
  const Matrix inputs_tranpose = transpose(inputs); 
  const Matrix weights_tranpose = transpose(weights); 
  Matrix delta;
  
  if (use_activation) {
      Matrix relu_grad = relu_backward(activations, grad);
      delta = std::move(relu_grad);
  } else {
    delta = std::move(grad);
  }
  
  grads = matmul(inputs_tranpose, delta) / (float)bsz;
  Matrix input_grads = matmul(delta, weights_tranpose); 
  return input_grads;
}

void SingleLayerPerceptron::update_weights(const float lr) {
  const Matrix update = grads * (-1 * lr);
  weights = weights + update;
}



MLP::MLP(int feat_dim, int num_layers) : feat_dim(feat_dim), num_layers(num_layers) {
    for (int i = 0; i < num_layers; ++i) {
        layers.push_back(SingleLayerPerceptron(feat_dim, feat_dim));
    }
    layers.push_back(SingleLayerPerceptron(output_classes, feat_dim, false));
}

Matrix MLP::forward(Matrix& input){
    Matrix* y = &layers[0].forward(input);
    for (int i = 1; i < num_layers; ++i) {
        y = &layers[i].forward(*y);
    }
    y = &layers[num_layers].forward(*y); // Classification head
    Matrix result = softmax(*y);
    return result;
}

void MLP::backward(const Matrix& labels, const Matrix& preds){
    Matrix grads = ce_softmax_bwd(labels, preds);
    for (int i = num_layers; i >= 0; --i) {
        grads = layers[i].backward(grads);
    }
}

void MLP::update_weights(const float lr) {
  for (int i = 0; i < num_layers + 1; ++i) {
      layers[i].update_weights(lr);
  }
}

void MLP::randomise(const unsigned long seed) {
    for (int i = 0; i < num_layers + 1; ++i) {
        layers[i].randomise(seed + i); 
    }
}
