#include "model.h"
#include "read_mnist.h"
#include "test_utils.h"

std::pair<Matrix, Matrix> construct_batch(
  std::vector<std::vector<unsigned char> >& images,
  std::vector<unsigned char>& labels,
  int bsz, 
  int step
) {
  int offset = step * bsz;

  // Prepare input 
  Matrix batch(28 * 28, bsz);
  float* data = new float[bsz * 28 * 28];
  for (int i = 0; i < bsz; ++i) {
    for (int j = 0; j < 28 * 28; ++j) {
      data[i * 28 * 28 + j] = (float)images[offset + i][j] / 255.0; // Normalise to [0, 1]
    }
  }
  batch.setData(data);

  // Prepare labels
  Matrix labels_batch(1, bsz);
  float* labels_data = new float[bsz];
  for (int i = 0; i < bsz; ++i) {
    labels_data[i] = (float)labels[offset + i];
  }
  labels_batch.setData(labels_data);

  return std::make_pair(batch, labels_batch);
}

void train_loop(
  MLP& mlp, 
  std::vector<std::vector<unsigned char> >& images,
  std::vector<unsigned char>& labels,
  int num_iters,
  int bsz,
  float lr
){
    float* data = new float[bsz * 28 * 28];
    Matrix output(28 * 28, bsz);
    for (int i = 0; i < num_iters; ++i) {
        std::cout << "Iteration " << i << std::endl;
        std::pair<Matrix, Matrix> data_and_labels = construct_batch(images, labels, bsz, i);
        Matrix output = mlp.forward(data_and_labels.first);
        output.getData(data);
        printMatrixData(data, 10, bsz, "for output tensor");
        float loss = output.get_ce_loss(data_and_labels.second).sum();

        data_and_labels.second.getData(data);
        printMatrixData(data, 1, bsz, "for labels tensor");
        std::cout << "Loss: " << loss << std::endl;
        mlp.backward(data_and_labels.second, output);
        mlp.update_weights(lr);
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
    int num_layers = 4;
    
    MLP mlp(feat_dim, num_layers);
    mlp.randomise(0);

    train_loop(mlp, train_images, train_labels, 1, 1, 0.001);

    return 0;
}
