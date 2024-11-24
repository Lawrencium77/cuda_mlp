#include <math.h>

template <typename T>
SingleLayerPerceptron<T>::SingleLayerPerceptron(const int dim_out, const int dim_in, const bool use_activation)
    : dim_out(dim_out),
      dim_in(dim_in),
      weights(dim_in, dim_out),
      grads(dim_in, dim_out),
      use_activation(use_activation) {}

template <typename T>
void SingleLayerPerceptron<T>::randomise(const unsigned long seed) {
    float max_f32 = 1.0f / sqrt(static_cast<float>(dim_in)); // Xavier initialisation
    float min_f32 = -max_f32;

    T max_T, min_T;
    if constexpr (std::is_same_v<T, half>) {
        max_T = __float2half(max_f32);
        min_T = __float2half(min_f32);
    } else {
        max_T = max_f32;
        min_T = min_f32;
    }

    weights.random(seed, min_T, max_T);
}

template <typename T>
Matrix<T>& SingleLayerPerceptron<T>::forward(Matrix<T>& input) {
  // weights: dim_in x dim_out
  // input: bsz x dim_in
  // output: bsz x dim_out

  // Using a ptr means we need to make sure that the input matrix is not modified between the forward and backward call.
  inputs = &input;

  Matrix<T> Z = matmul(input, weights);
  if (use_activation) {
    activations = relu(Z);
  } else {
    activations = std::move(Z);
  }
  return activations;
}

template <typename T>
Matrix<T> SingleLayerPerceptron<T>::backward(Matrix<T>& grad) {
  // weights: dim_in x dim_out
  // grad: bsz x dim_out
  // input_grads: bsz x dim_in
  // weight_grads: dim_in x dim_out
  const int bsz = grad.rows;
  const Matrix<T> inputs_tranpose = transpose(*inputs);
  const Matrix<T> weights_tranpose = transpose(weights);
  Matrix<T> delta;

  if (use_activation) {
      Matrix<T> relu_grad = relu_backward(activations, grad);
      delta = std::move(relu_grad);
  } else {
    delta = std::move(grad);
  }

  grads = matmul(inputs_tranpose, delta) / (float)bsz;
  Matrix<T> input_grads = matmul(delta, weights_tranpose);
  return input_grads;
}

template <typename T>
void SingleLayerPerceptron<T>::update_weights(const float lr) {
  const Matrix<T> update = grads * (-1 * lr);
  weights = weights + update;
}

template <typename T>
MLP<T>::MLP(int feat_dim, int num_layers) : feat_dim(feat_dim), num_layers(num_layers) {
    for (int i = 0; i < num_layers; ++i) {
        layers.push_back(SingleLayerPerceptron<T>(feat_dim, feat_dim));
    }
    layers.push_back(SingleLayerPerceptron<T>(output_classes, feat_dim, false));
}

template <typename T>
Matrix<T> MLP<T>::forward(Matrix<T>& input){
    Matrix<T>* y = &layers[0].forward(input);
    for (int i = 1; i < num_layers; ++i) {
        y = &layers[i].forward(*y);
    }
    y = &layers[num_layers].forward(*y); // Classification head
    Matrix<T> result = softmax(*y);
    return result;
}

template <typename T>
void MLP<T>::backward(const Matrix<T>& labels, const Matrix<T>& preds){
    Matrix<T> grads = ce_softmax_bwd(labels, preds);
    for (int i = num_layers; i >= 0; --i) {
        grads = layers[i].backward(grads);
    }
}

template <typename T>
void MLP<T>::update_weights(const float lr) {
  for (int i = 0; i < num_layers + 1; ++i) {
      layers[i].update_weights(lr);
  }
}

template <typename T>
void MLP<T>::randomise(const unsigned long seed) {
    for (int i = 0; i < num_layers + 1; ++i) {
        layers[i].randomise(seed + i);
    }
}
