#include "matrix.h"
#include "vector.h"
#include <vector>

class SingleLayerPerceptron {
  private:
    int feat_dim;
    Matrix weights;

  public:
    SingleLayerPerceptron(int feat_dim);
    Matrix forward(Matrix& input);
};



class MLP {
  
  private:
    int feat_dim;
    int num_layers; 
    std::vector<SingleLayerPerceptron> layers;
  
  public:
    MLP(int feat_dim, int num_layers);
    Matrix forward(Matrix& input);

};
