#include <iostream>

// Assuming matrix has shape (bsz, feats)
void printMatrixData(const float* data, int bsz, int feats, std::string message = "") {
    std::cout << "Data " << message << ": \n";
    for (int i = 0; i < bsz; i++){
        for (int j = 0; j < feats; j++){
            std::cout << data[i * feats + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
};
