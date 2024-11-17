#include "config_reader.h"
#include "cuda_utils.h"
#include "model.h"
#include "read_mnist.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
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

    Matrix image_batch(current_bsz, feat_dim);
    float* image_data = new float[current_bsz * feat_dim];
    for (int i = 0; i < current_bsz; ++i) {
        int idx = offset + i;
        for (int j = 0; j < feat_dim; ++j) {
            image_data[i * feat_dim + j] = static_cast<float>(images[idx][j]) / 255.0f; // Normalize
        }
    }
    image_batch.setHostData(image_data);
    image_batch.toDevice();

    Matrix labels_batch(current_bsz, 1);
    float* labels_data = new float[current_bsz];
    for (int i = 0; i < current_bsz; ++i) {
        int idx = offset + i;
        labels_data[i] = static_cast<float>(labels[idx]);
    }
    labels_batch.setHostData(labels_data);
    labels_batch.toDevice();

    return std::make_pair(std::move(image_batch), std::move(labels_batch));
}

std::pair<float, float> get_val_stats(
  MLP& mlp,
  const std::vector<std::vector<unsigned char>>& val_images,
  const std::vector<unsigned char>& val_labels,
  const int bsz,
  const int feat_dim
) {
  const int num_samples = val_images.size();
  const int num_batches = (num_samples + bsz - 1) / bsz;
  float total_loss = 0.0f;
  float total_correct = 0.0f;

  for (int i = 0; i < num_batches; ++i) {
    const int current_bsz = std::min(bsz, num_samples - i * bsz);
    std::pair<Matrix, Matrix> data_and_labels = prepare_batch(val_images, val_labels, current_bsz, feat_dim, i);
    Matrix output = mlp.forward(data_and_labels.first);
    std::pair<Matrix, Matrix> loss_and_preds = get_ce_loss_and_accuracy(output, data_and_labels.second);
    total_loss += matsum(loss_and_preds.first);
    total_correct += matsum(loss_and_preds.second);
  }
  return std::make_pair(total_loss / num_samples, 100 * total_correct / num_samples);
}

std::vector<std::vector<float>> train_loop(
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
  const bool time_epoch
){
    std::vector<float> train_losses;
    std::vector<float> val_losses;
    std::vector<float> val_accs;

    const int num_samples = train_images.size();
    const int num_batches_per_epoch = (num_samples + bsz - 1) / bsz;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        std::cout << "\nEpoch " << epoch + 1 << "/" << num_epochs << std::endl;

        shuffle_dataset(train_images, train_labels);

        auto start_time = std::chrono::high_resolution_clock::now();
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
            mlp.backward(data_and_labels.second, output);
            mlp.update_weights(lr);

            cudaDeviceSynchronize();
            CHECK_CUDA_STATE();

            float loss = matsum(get_ce_loss(output, data_and_labels.second)) / current_bsz;
            train_losses.push_back(loss);

            if (verbose) {
                std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "] "
                          << "Batch [" << batch_idx + 1 << "/" << num_batches_per_epoch << "] "
                          << "Loss: " << loss << std::endl;
            }

        }

        // Validate at end of each epoch
        std::pair<float, float> val_loss_and_acc = get_val_stats(mlp, val_images, val_labels, bsz, feat_dim);
        val_losses.push_back(val_loss_and_acc.first);
        val_accs.push_back(val_loss_and_acc.second);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        if (time_epoch) {
            std::cout << "Epoch " << epoch << " took " << elapsed_time.count() << " ms" << std::endl;
        }

        std::cout << "Validation Loss after epoch " << epoch + 1 << ": " << val_loss_and_acc.first << std::endl;
        std::cout << "Validation Acc after epoch " << epoch + 1 << ": " << val_loss_and_acc.second << "%" << std::endl;
    }

    std::vector<std::vector<float>> metrics = {train_losses, val_losses, val_accs};
    return metrics;
}

void save_metric(const std::vector<float>& losses, const std::string filename) {
    std::ofstream loss_file(filename);
    for (const auto& l : losses) {
        loss_file << l << "\n";
    }
    loss_file.close();
}

void train(const std::string config_file) {
    std::map<std::string, std::string> config = read_config(config_file);

    const std::string data_dir = config["data_dir"];
    const std::string train_image_file = data_dir + "/train-images-idx3-ubyte";
    const std::string val_image_file = data_dir + "/t10k-images-idx3-ubyte";
    const std::string train_label_file = data_dir + "/train-labels-idx1-ubyte";
    const std::string val_label_file = data_dir + "/t10k-labels-idx1-ubyte";

    std::vector<std::vector<unsigned char> > train_images = read_mnist_images(train_image_file);
    std::vector<std::vector<unsigned char> > val_images = read_mnist_images(val_image_file);
    std::vector<unsigned char> train_labels = read_mnist_labels(train_label_file);
    std::vector<unsigned char> val_labels = read_mnist_labels(val_label_file);

    const std::string log_dir = config["log_dir"];
    std::filesystem::create_directory(log_dir);

    MLP mlp(std::stoi(config["feat_dim"]), std::stoi(config["num_layers"]));
    mlp.randomise(0);
    cudaDeviceSynchronize();

    std::vector<std::vector<float>> metrics = train_loop(
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
      std::stoi(config["time_epoch"])
    );

    save_metric(metrics[0], log_dir + "/train_losses.txt");
    save_metric(metrics[1], log_dir + "/val_losses.txt");
    save_metric(metrics[2], log_dir + "/val_accs.txt");
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file>\n";
        return EXIT_FAILURE;
    }
    train(argv[1]);
    Matrix::allocator.cleanup();
    return 0;
}
