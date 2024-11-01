#include "matrix.h"
#include <vector>

struct SingleLayerPerceptron {
    int dim_out;
    int dim_in;
    Matrix weights;
    Matrix grads;
    Matrix inputs;
    Matrix activations;
    bool sigmoid;

    SingleLayerPerceptron(int dim_out, int dim_in, bool sigmoid = true);
    Matrix forward(Matrix& input);
    Matrix backward(Matrix& grad);
    void update_weights(float lr);
    void randomise(unsigned long seed = 0);
};



struct MLP {
    int feat_dim;
    int num_layers; 
    int output_classes = 10;
    std::vector<SingleLayerPerceptron> layers;

    MLP(int feat_dim, int num_layers);
    Matrix forward(Matrix& input);
    void backward(Matrix& labels, Matrix& preds);
    void update_weights(float lr);
    void randomise(unsigned long seed = 0);
};
