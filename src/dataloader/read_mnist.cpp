#include <iostream>
#include <fstream>
#include <vector>
#include <string>

int readBigEndianInt(std::ifstream& file) {
    unsigned char bytes[4];
    if (!file.read(reinterpret_cast<char*>(bytes), 4)) {
        std::cerr << "Error reading integer from file.\n";
        exit(EXIT_FAILURE);
    }
    return (int)((bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3]);
}

std::vector<std::vector<unsigned char> > read_mnist_images(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    int magic_number = readBigEndianInt(file);
    if (magic_number != 2051) {
        std::cerr << "Invalid MNIST image file!\n";
        exit(EXIT_FAILURE);
    }

    int number_of_images = readBigEndianInt(file);
    int n_rows = readBigEndianInt(file);
    int n_cols = readBigEndianInt(file);

    std::vector<std::vector<unsigned char> > images(number_of_images, std::vector<unsigned char>(n_rows * n_cols));

    for (int i = 0; i < number_of_images; ++i) {
        if (!file.read(reinterpret_cast<char*>(images[i].data()), n_rows * n_cols)) {
            std::cerr << "Error reading image data.\n";
            exit(EXIT_FAILURE);
        }
    }

    file.close();
    return images;
}

std::vector<unsigned char> read_mnist_labels(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    int magic_number = readBigEndianInt(file);
    if (magic_number != 2049) {
        std::cerr << "Invalid MNIST label file!\n";
        exit(EXIT_FAILURE);
    }

    int number_of_items = readBigEndianInt(file);
    std::vector<unsigned char> labels(number_of_items);

    if (!file.read(reinterpret_cast<char*>(labels.data()), number_of_items)) {
        std::cerr << "Error reading label data.\n";
        exit(EXIT_FAILURE);
    }

    file.close();
    return labels;
}

int main() {
    std::string image_file = "/Users/latkins/git/cuda_mlp/mnist_data/t10k-images-idx3-ubyte";
    std::string label_file = "/Users/latkins/git/cuda_mlp/mnist_data/t10k-labels-idx1-ubyte";

    std::vector<std::vector<unsigned char> > images = read_mnist_images(image_file);
    std::vector<unsigned char> labels = read_mnist_labels(label_file);

    return 0;
}
