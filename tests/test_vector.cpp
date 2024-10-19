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

void testAdd(float* data, int size) {
    Vector vector1(size);
    Vector vector2(size);
    vector1.setData(data);
    vector2.setData(data);

    Vector new_vector = vector1 + vector2;

    new_vector.getData(data);
    printData(data, size);
}

void testSub(float* data, int size) {
    Vector vector1(size);
    Vector vector2(size);
    vector1.setData(data);
    vector2.setData(data);

    Vector new_vector = vector1 - vector2;

    new_vector.getData(data);
    printData(data, size);
}

void testMul(float* data, int size) {
    Vector vector1(size);
    Vector vector2(size);
    vector1.setData(data);
    vector2.setData(data);

    Vector new_vector = vector1 * vector2;

    new_vector.getData(data);
    printData(data, size);
}

void testDiv(float* data, int size) {
    Vector vector1(size);
    Vector vector2(size);
    vector1.setData(data);
    vector2.setData(data);

    Vector new_vector = vector1 / vector2;

    new_vector.getData(data);
    printData(data, size);
}

void testSigmoid(float* data, int size) {
    Vector my_vector(size);
    my_vector.setData(data);

    Vector new_vector = my_vector.sigmoid();

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
    
    std::cout << "Testing Sigmoid Op" << std::endl;
    testSigmoid(data, size);

    std::cout << "\nTesting Const Ops" << std::endl;
    testAddConst(data, value, size);
    testSubConst(data, value, size);
    testMulConst(data, value, size);
    testDivConst(data, value, size);

    std::cout << "\nTesting Vector-Vector ops" << std::endl;
    testDiv(data, size); // Test div first to avoid divide by zero
    testAdd(data, size);
    testSub(data, size);
    testMul(data, size);

    delete[] data;
}

int main() {
    runTests();
    return 0;
}
