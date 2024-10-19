#include <iostream>
#include "vector.h"

void printData(const float* data, int size) {
    std::cout << "Data: ";
    for (int i = 0; i < size; i++){
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
};

void testAddConst(float* data, float value, int size) {
    Vector my_vector(size);
    my_vector.setData(data);

    Vector new_vector = my_vector + value;

    new_vector.getData(data);
    printData(data, size);
}

void testSubConst(float* data, float value, int size) {
    Vector my_vector(size);
    my_vector.setData(data);

    Vector new_vector = my_vector - value;

    new_vector.getData(data);
    printData(data, size);
}

void testMulConst(float* data, float value, int size) {
    Vector my_vector(size);
    my_vector.setData(data);

    Vector new_vector = my_vector * value;

    new_vector.getData(data);
    printData(data, size);
}

void testDivConst(float* data, float value, int size) {
    Vector my_vector(size);
    my_vector.setData(data);

    Vector new_vector = my_vector / value;

    new_vector.getData(data);
    printData(data, size);
}

void runTests() {
    int size = 5;
    float value = 10.0;
    float* data = new float[size];
    for (int i = 0; i < size; i++){
        data[i] = 1.0f;
    }
    testAddConst(data, value, size);
    testSubConst(data, value, size);
    testMulConst(data, value, size);
    testDivConst(data, value, size);
    delete[] data;
}

int main() {
    runTests();
    return 0;
}
