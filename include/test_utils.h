#include <iostream>

void printMatrixData(const float* data, int rows, int cols, std::string message = "") {
    std::cout << "Data " << message << ": \n";
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            std::cout << data[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
};
