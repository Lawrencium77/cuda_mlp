#include "config_reader.h"
#include "model.h"
#include "read_mnist.h"
#include "test_utils.h"
#include <filesystem>

std::pair<Matrix, Matrix> prepare_batch(
  std::vector<std::vector<unsigned char> >& images,
  std::vector<unsigned char>& labels,
  int bsz, 
  int step
) {
  int offset = step * bsz;

  // Prepare input 
  Matrix batch(bsz, 28 * 28);
  float* data = new float[bsz * 28 * 28];
  for (int i = 0; i < bsz; ++i) {
    for (int j = 0; j < 28 * 28; ++j) {
      data[i * 28 * 28 + j] = (float)images[offset + i][j] / 255.0; // Normalise
    }
  }
  batch.setData(data);

  // Prepare labels
  Matrix labels_batch(bsz, 1);
  float* labels_data = new float[bsz];
  for (int i = 0; i < bsz; ++i) {
    labels_data[i] = (float)labels[offset + i];
  }
  labels_batch.setData(labels_data);

  return std::make_pair(batch, labels_batch);
}

float get_val_loss(
  MLP& mlp,
  std::vector<std::vector<unsigned char>>& val_images,
  std::vector<unsigned char>& val_labels,
  int bsz
) {
  int num_samples = val_images.size();
  int num_batches = (num_samples + bsz - 1) / bsz;
  float total_loss = 0.0f;

  for (int i = 0; i < num_batches; ++i) {
    int current_bsz = std::min(bsz, num_samples - i * bsz);
    std::pair<Matrix, Matrix> data_and_labels = prepare_batch(val_images, val_labels, current_bsz, i);
    Matrix output = mlp.forward(data_and_labels.first);
    float loss = output.get_ce_loss(data_and_labels.second).sum();
    total_loss += loss;
  }
  return total_loss / num_samples;
}

std::pair<std::vector<float>, std::vector<float>> train_loop(
  MLP& mlp, 
  std::vector<std::vector<unsigned char>>& train_images,
  std::vector<unsigned char>& train_labels,
  std::vector<std::vector<unsigned char>>& val_images,
  std::vector<unsigned char>& val_labels,
  const int num_iters,
  const int bsz,
  const float lr,
  const int val_every,
  const std::string& log_dir
){
    std::vector<float> train_losses;
    std::vector<float> val_losses;

    for (int step = 0; step < num_iters; ++step) {
        std::pair<Matrix, Matrix> data_and_labels = prepare_batch(train_images, train_labels, bsz, step);
        
        Matrix output = mlp.forward(data_and_labels.first);
        float loss = output.get_ce_loss(data_and_labels.second).sum() / bsz;
        train_losses.push_back(loss);

        mlp.backward(data_and_labels.second, output);
        mlp.update_weights(lr);

        if (step % val_every == 0) {
            float val_loss = get_val_loss(mlp, val_images, val_labels, bsz);
            std::cout << "Step " << step <<  " Validation Loss: " << val_loss << std::endl;
            val_losses.push_back(val_loss);
        }
    }
    
    return std::make_pair(train_losses, val_losses);
}

void save_losses(std::vector<float>& losses, std::string filename) {
    std::ofstream loss_file(filename);
    for (const auto& l : losses) {
        loss_file << l << "\n";
    }
    loss_file.close();
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file>\n";
        return EXIT_FAILURE;
    }

    // Load config
    std::string config_file = argv[1];
    std::map<std::string, std::string> config = read_config(config_file);
    const std::string data_dir = config["data_dir"];
    const std::string log_dir = config["log_dir"];
    const int num_layers = std::stoi(config["num_layers"]);
    const int val_every = std::stoi(config["val_every"]);
    
    std::filesystem::create_directory(log_dir);

    std::string train_image_file = data_dir + "/train-images-idx3-ubyte";
    std::string val_image_file = data_dir + "/t10k-images-idx3-ubyte";
    std::string train_label_file = data_dir + "/train-labels-idx1-ubyte";
    std::string val_label_file = data_dir + "/t10k-labels-idx1-ubyte";

    // Load all data into memory. We split it into batches on the fly.
    std::vector<std::vector<unsigned char> > train_images = read_mnist_images(train_image_file);
    std::vector<std::vector<unsigned char> > val_images = read_mnist_images(val_image_file);
    std::vector<unsigned char> train_labels = read_mnist_labels(train_label_file);
    std::vector<unsigned char> val_labels = read_mnist_labels(val_label_file);

    const int feat_dim = 28 * 28;
    
    MLP mlp(feat_dim, num_layers);
    mlp.randomise(0);

    std::pair<std::vector<float>, std::vector<float>> losses = train_loop(mlp, train_images, train_labels, val_images, val_labels, 1600, 8, 0.01f, val_every, log_dir);
    
    save_losses(losses.first, log_dir + "/train_losses.txt");
    save_losses(losses.second, log_dir + "/val_losses.txt");

    return 0;
}
