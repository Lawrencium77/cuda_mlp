#include "model.h"

SingleLayerPerceptron::SingleLayerPerceptron(int dim_out, int dim_in) : dim_out(dim_out), dim_in(dim_in), weights(dim_out, dim_in) {}

void SingleLayerPerceptron::randomise(unsigned long seed) {
  weights.random(seed);
}

Matrix SingleLayerPerceptron::forward(Matrix& input) {
  // weights: dim_out x dim_in
  // input: dim_in x bsz
  // output: dim_out x bsz
  Matrix intermediate = weights.matmul(input); 
  return intermediate.sigmoid();
}


MLP::MLP(int feat_dim, int num_layers) : feat_dim(feat_dim), num_layers(num_layers) {
    for (int i = 0; i < num_layers; ++i) {
        layers.push_back(SingleLayerPerceptron(feat_dim, feat_dim));
    }
    layers.push_back(SingleLayerPerceptron(output_classes, feat_dim));
}

Matrix MLP::forward(Matrix& input){
    Matrix y = input;
    for (int i = 0; i < num_layers; ++i) {
        y = layers[i].forward(y);
    }
    y = layers[num_layers].forward(y); // Classification head
    return y;
}

void MLP::randomise(unsigned long seed) {
    for (int i = 0; i < num_layers + 1; ++i) { // Randomise each layer, and classification head
        layers[i].randomise(seed + i); 
    }
}
