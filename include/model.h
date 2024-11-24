// TODO: Implement PyTorch mixed-precision recipe:
//  * FP32 weights 
//  * FP32 weight updates
//  * FP16 grads/activations
// Current we either do FP32 or FP16 for everything
#include "matrix.h"
#include <vector>

template <typename T>
struct SingleLayerPerceptron {
    const int dim_out;
    const int dim_in;
    Matrix<T> weights;
    Matrix<T> grads;
    Matrix<T>* inputs;
    Matrix<T> activations;
    const bool use_activation;

    SingleLayerPerceptron(const int dim_out, const int dim_in, const bool use_activation = true);
    Matrix<T>& forward(Matrix<T>& input);
    Matrix<T> backward(Matrix<T>& grad);
    void update_weights(const float lr);
    void randomise(const unsigned long seed = 0);
};



template <typename T>
struct MLP {
    const int feat_dim;
    const int num_layers; 
    const int output_classes = 10;
    std::vector<SingleLayerPerceptron<T>> layers;

    MLP(int feat_dim, int num_layers);
    Matrix<T> forward(Matrix<T>& input);
    void backward(const Matrix<T>& labels, const Matrix<T>& preds);
    void update_weights(const float lr);
    void randomise(const unsigned long seed = 0);
};

#include "model.cuh"
