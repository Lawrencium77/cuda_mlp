GCC = g++
NVCC = nvcc

CFLAGS = -I./include

SRC_DIR = src
TEST_DIR = tests
BUILD_DIR = build

DATA_SRC_FILES = $(SRC_DIR)/dataloader/read_mnist.cpp
VECTOR_SRC_FILES = $(SRC_DIR)/vector/vector.cu $(SRC_DIR)/vector/vector_kernels.cu
MATRIX_SRC_FILES = $(SRC_DIR)/matrix/matrix.cu $(SRC_DIR)/matrix/matrix_kernels.cu

VECTOR_TEST_FILE = $(TEST_DIR)/test_vector.cpp
MATRIX_TEST_FILE = $(TEST_DIR)/test_matrix.cpp

DATA_OUTPUT = $(BUILD_DIR)/read_mnist
VECTOR_OUTPUT = $(BUILD_DIR)/test_vector
MATRIX_OUTPUT = $(BUILD_DIR)/test_matrix

all: $(VECTOR_OUTPUT) $(MATRIX_OUTPUT) $(DATA_OUTPUT)

data: $(DATA_OUTPUT)

$(DATA_OUTPUT): $(DATA_SRC_FILES)
	mkdir -p $(BUILD_DIR)
	$(GCC) $(CFLAGS) -o $(DATA_OUTPUT) $(DATA_SRC_FILES)

$(VECTOR_OUTPUT): $(VECTOR_SRC_FILES) $(VECTOR_TEST_FILE)
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(CFLAGS) -o $(VECTOR_OUTPUT) $(VECTOR_TEST_FILE) $(VECTOR_SRC_FILES)

$(MATRIX_OUTPUT): $(MATRIX_SRC_FILES) $(MATRIX_TEST_FILE)
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(CFLAGS) -o $(MATRIX_OUTPUT) $(MATRIX_TEST_FILE) $(MATRIX_SRC_FILES)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
