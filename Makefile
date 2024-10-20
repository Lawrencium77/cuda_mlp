NVCC = nvcc

CFLAGS = -I./include

SRC_DIR = src
TEST_DIR = tests
BUILD_DIR = build

VECTOR_SRC_FILES = $(SRC_DIR)/vector/vector.cu $(SRC_DIR)/vector/vector_kernels.cu
MATRIX_SRC_FILES = $(SRC_DIR)/matrix/matrix.cu $(SRC_DIR)/matrix/matrix_kernels.cu

VECTOR_TEST_FILE = $(TEST_DIR)/test_vector.cpp
MATRIX_TEST_FILE = $(TEST_DIR)/test_matrix.cpp

VECTOR_OUTPUT = $(BUILD_DIR)/test_vector
MATRIX_OUTPUT = $(BUILD_DIR)/test_matrix

all: $(VECTOR_OUTPUT) $(MATRIX_OUTPUT)

$(VECTOR_OUTPUT): $(VECTOR_SRC_FILES) $(VECTOR_TEST_FILE)
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(CFLAGS) -o $(VECTOR_OUTPUT) $(VECTOR_TEST_FILE) $(VECTOR_SRC_FILES)

$(MATRIX_OUTPUT): $(MATRIX_SRC_FILES) $(MATRIX_TEST_FILE)
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(CFLAGS) -o $(MATRIX_OUTPUT) $(MATRIX_TEST_FILE) $(MATRIX_SRC_FILES)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
