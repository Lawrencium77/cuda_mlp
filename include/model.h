#include "matrix.h"
#include <vector>

constexpr int IMAGE_FEAT_DIM = 28 * 28;

struct SingleLayerPerceptron {
  const int dim_out;
  const int dim_in;
  Matrix weights;
  Matrix grads;
  Matrix *inputs;
  Matrix activations;
  const bool use_activation;

  SingleLayerPerceptron(const int dim_out, const int dim_in,
                        const bool use_activation = true);
  Matrix &forward(Matrix &input);
  Matrix backward(Matrix &grad);
  void update_weights(const float lr);
  void randomise(const unsigned long seed = 0);
};

struct MLP {
  const int input_dim;
  const int feat_dim;
  const int num_layers;
  const int output_classes = 10;
  std::vector<SingleLayerPerceptron> layers;

  MLP(int feat_dim, int num_layers);
  Matrix forward(Matrix &input);
  void backward(const Matrix &labels, const Matrix &preds);
  void update_weights(const float lr);
  void randomise(const unsigned long seed = 0);
};
