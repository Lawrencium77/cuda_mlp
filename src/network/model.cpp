#include "model.h"

SingleLayerPerceptron::SingleLayerPerceptron(int feat_dim) : feat_dim(feat_dim), weights(feat_dim, feat_dim) {}

Matrix SingleLayerPerceptron::forward(Matrix& input) {
  Matrix intermediate = weights.matmul(input);
  return intermediate.sigmoid();
}


MLP::MLP(int feat_dim, int num_layers) : feat_dim(feat_dim), num_layers(num_layers) {
    for (int i = 0; i < num_layers; ++i) {
        layers.push_back(SingleLayerPerceptron(feat_dim));
    }
}

Matrix MLP::forward(Matrix& input){
    Matrix y = input;
    for (int i = 0; i < num_layers; ++i) {
        y = layers[i].forward(y);
    }
    return y;
}
