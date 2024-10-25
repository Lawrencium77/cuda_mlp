#include "matrix.h"
#include "vector.h"
#include <vector>

class SingleLayerPerceptron {
  private:
    int dim_out;
    int dim_in;
    Matrix weights;

  public:
    SingleLayerPerceptron(int dim_out, int dim_in);
    Matrix forward(Matrix& input);
    void randomise(unsigned long seed = 0);
};



class MLP {
  
  private:
    int feat_dim;
    int num_layers; 
    std::vector<SingleLayerPerceptron> layers;
  
  public:
    MLP(int feat_dim, int num_layers);
    Matrix forward(Matrix& input);
    void randomise(unsigned long seed = 0);

};
