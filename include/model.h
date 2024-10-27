#include "matrix.h"
#include <vector>

class SingleLayerPerceptron {
  private:
    int dim_out;
    int dim_in;
    Matrix weights;
    Matrix grads;
    Matrix inputs;
    bool sigmoid;

  public:
    SingleLayerPerceptron(int dim_out, int dim_in, bool sigmoid = true);
    Matrix forward(Matrix& input);
    Matrix backward(Matrix& grad);
    void randomise(unsigned long seed = 0);
};



class MLP {
  
  private:
    int feat_dim;
    int num_layers; 
    int output_classes = 10;
    std::vector<SingleLayerPerceptron> layers;
  
  public:
    MLP(int feat_dim, int num_layers);
    Matrix forward(Matrix& input);
    void backward(Matrix& labels, Matrix& preds);
    void randomise(unsigned long seed = 0);

};
