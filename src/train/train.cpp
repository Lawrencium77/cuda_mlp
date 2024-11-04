#include "config_reader.h"
#include "model.h"
#include "read_mnist.h"
#include "test_utils.h"
#include <filesystem>
#include <algorithm>
#include <random>

void shuffle_dataset(
    std::vector<std::vector<unsigned char>>& images,
    std::vector<unsigned char>& labels
) {
    std::vector<size_t> indices(images.size());
    for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    std::vector<std::vector<unsigned char>> images_shuffled(images.size());
    std::vector<unsigned char> labels_shuffled(labels.size());

    for (size_t i = 0; i < indices.size(); ++i) {
        images_shuffled[i] = images[indices[i]];
        labels_shuffled[i] = labels[indices[i]];
    }

    images = std::move(images_shuffled);
    labels = std::move(labels_shuffled);
}

std::pair<Matrix, Matrix> prepare_batch(
  const std::vector<std::vector<unsigned char> >& images,
  const std::vector<unsigned char>& labels,
  const int current_bsz,
  const int feat_dim, 
  const int batch_idx
) {
    const int offset = batch_idx * current_bsz;

    // Prepare input
    Matrix batch(current_bsz, feat_dim);
    float* data = new float[current_bsz * feat_dim];
    for (int i = 0; i < current_bsz; ++i) {
        int idx = offset + i;
        for (int j = 0; j < feat_dim; ++j) {
            data[i * feat_dim + j] = static_cast<float>(images[idx][j]) / 255.0f; // Normalize
        }
    }
    batch.setData(data);

    // Prepare labels
    Matrix labels_batch(current_bsz, 1);
    float* labels_data = new float[current_bsz];
    for (int i = 0; i < current_bsz; ++i) {
        int idx = offset + i;
        labels_data[i] = static_cast<float>(labels[idx]);
    }
    labels_batch.setData(labels_data);

    return std::make_pair(batch, labels_batch);
}

float get_val_loss(
  MLP& mlp,
  const std::vector<std::vector<unsigned char>>& val_images,
  const std::vector<unsigned char>& val_labels,
  const int bsz,
  const int feat_dim
) {
  const int num_samples = val_images.size();
  const int num_batches = (num_samples + bsz - 1) / bsz;
  float total_loss = 0.0f;

  for (int i = 0; i < num_batches; ++i) {
    const int current_bsz = std::min(bsz, num_samples - i * bsz);
    std::pair<Matrix, Matrix> data_and_labels = prepare_batch(val_images, val_labels, current_bsz, feat_dim, i);
    Matrix output = mlp.forward(data_and_labels.first);
    float loss = matsum(get_ce_loss(output, data_and_labels.second));
    total_loss += loss;
  }
  return total_loss / num_samples;
}

std::pair<std::vector<float>, std::vector<float>> train_loop(
  MLP& mlp, 
  std::vector<std::vector<unsigned char>>& train_images,
  std::vector<unsigned char>& train_labels,             
  const std::vector<std::vector<unsigned char>>& val_images,
  const std::vector<unsigned char>& val_labels,
  const int num_epochs,
  const int bsz,
  const int feat_dim,
  const float lr,
  const bool verbose,
  const std::string& log_dir
){
    std::vector<float> train_losses;
    std::vector<float> val_losses;

    const int num_samples = train_images.size();
    const int num_batches_per_epoch = (num_samples + bsz - 1) / bsz;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        std::cout << "Epoch " << epoch + 1 << "/" << num_epochs << std::endl;

        shuffle_dataset(train_images, train_labels);

        for (int batch_idx = 0; batch_idx < num_batches_per_epoch; ++batch_idx) {
            const int current_bsz = std::min(bsz, num_samples - batch_idx * bsz);
            std::pair<Matrix, Matrix> data_and_labels = prepare_batch(
                train_images,
                train_labels,
                current_bsz,
                feat_dim,
                batch_idx
            );

            Matrix output = mlp.forward(data_and_labels.first);
            float loss = matsum(get_ce_loss(output, data_and_labels.second)) / current_bsz;
            train_losses.push_back(loss);

            if (verbose) {
                std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "] "
                          << "Batch [" << batch_idx + 1 << "/" << num_batches_per_epoch << "] "
                          << "Loss: " << loss << std::endl;
            }

            mlp.backward(data_and_labels.second, output);
            mlp.update_weights(lr);
        }

        // Validate at end of each epoch
        float val_loss = get_val_loss(mlp, val_images, val_labels, bsz, feat_dim);
        val_losses.push_back(val_loss);
        std::cout << "Validation Loss after epoch " << epoch + 1 << ": " << val_loss << std::endl;
    }
    
    return std::make_pair(train_losses, val_losses);
}

void save_losses(const std::vector<float>& losses, const std::string filename) {
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
    const std::string config_file = argv[1];
    std::map<std::string, std::string> config = read_config(config_file);

    const std::string data_dir = config["data_dir"];
    const std::string train_image_file = data_dir + "/train-images-idx3-ubyte";
    const std::string val_image_file = data_dir + "/t10k-images-idx3-ubyte";
    const std::string train_label_file = data_dir + "/train-labels-idx1-ubyte";
    const std::string val_label_file = data_dir + "/t10k-labels-idx1-ubyte";

    // Load all data into memory. 
    std::vector<std::vector<unsigned char> > train_images = read_mnist_images(train_image_file);
    std::vector<std::vector<unsigned char> > val_images = read_mnist_images(val_image_file);
    std::vector<unsigned char> train_labels = read_mnist_labels(train_label_file);
    std::vector<unsigned char> val_labels = read_mnist_labels(val_label_file);
    
    const std::string log_dir = config["log_dir"];
    std::filesystem::create_directory(log_dir);

    MLP mlp(std::stoi(config["feat_dim"]), std::stoi(config["num_layers"]));
    mlp.randomise(0);

    std::pair<std::vector<float>, std::vector<float>> losses = train_loop(
      mlp, 
      train_images, 
      train_labels, 
      val_images, 
      val_labels, 
      std::stoi(config["num_epochs"]),
      std::stoi(config["bsz"]), 
      std::stoi(config["feat_dim"]), 
      std::stof(config["learning_rate"]), 
      std::stoi(config["verbose"]),
      log_dir
    );
    
    save_losses(losses.first, log_dir + "/train_losses.txt");
    save_losses(losses.second, log_dir + "/val_losses.txt");

    return 0;
}
