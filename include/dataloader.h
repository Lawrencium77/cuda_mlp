#include <iostream>
#include <fstream>
#include <vector>
#include <string>

std::vector<std::vector<std::uint8_t> > read_mnist_images(const std::string& filename);
std::vector<std::uint8_t> read_mnist_labels(const std::string& filename);
