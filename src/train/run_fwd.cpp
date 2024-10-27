#include "model.h"
#include "read_mnist.h"
#include "test_utils.h"

Matrix construct_batch(std::vector<std::vector<unsigned char> >& images, int bsz, int step) {
  Matrix batch(28 * 28, bsz);
  float* data = new float[bsz * 28 * 28];
  int offset = step * bsz;
  for (int i = 0; i < bsz; ++i) {
    for (int j = 0; j < 28 * 28; ++j) {
      data[i * 28 * 28 + j] = (float)images[offset + i][j] / 255.0; // Normalise to [0, 1]
    }
  }
  batch.setData(data);
  return batch;
}

void do_fwd_passes(MLP& mlp, std::vector<std::vector<unsigned char> >& images, int num_iters, int bsz){
    float* data = new float[bsz * 28 * 28];
    Matrix output(28 * 28, bsz);
    for (int i = 0; i < num_iters; ++i) {
        std::cout << "Iteration " << i << std::endl;
        Matrix input = construct_batch(images, bsz, i);
        output = mlp.forward(input);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <mnist_data_directory>\n";
        return EXIT_FAILURE;
    }

    std::string data_dir = argv[1];
    std::string train_image_file = data_dir + "/train-images-idx3-ubyte";
    std::string val_image_file = data_dir + "/t10k-images-idx3-ubyte";
    std::string train_label_file = data_dir + "/train-labels-idx1-ubyte";
    std::string val_label_file = data_dir + "/t10k-labels-idx1-ubyte";

    std::vector<std::vector<unsigned char> > train_images = read_mnist_images(train_image_file);
    std::vector<std::vector<unsigned char> > val_images = read_mnist_images(val_image_file);
    std::vector<unsigned char> train_labels = read_mnist_labels(train_label_file);
    std::vector<unsigned char> val_labels = read_mnist_labels(val_label_file);

    int feat_dim = 28 * 28;
    int num_layers = 10;
    
    MLP mlp(feat_dim, num_layers);
    mlp.randomise(0);

    do_fwd_passes(mlp, train_images, 10, 8);

    return 0;
}
