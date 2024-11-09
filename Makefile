GCC = g++
NVCC = nvcc

CFLAGS = -I./include
ifeq ($(DEBUG), 1)
    CFLAGS += -g
endif

SRC_DIR = src
TEST_DIR = tests
BUILD_DIR = build

DATA_SRC_FILES = $(SRC_DIR)/dataloader/read_mnist.cpp
MATRIX_SRC_FILES = $(SRC_DIR)/matrix/matrix.cu $(SRC_DIR)/matrix/matrix_kernels.cu
MODEL_SRC_FILES = $(SRC_DIR)/model/model.cpp
TRAINING_SRC_FILES = $(SRC_DIR)/train/config_reader.cpp $(SRC_DIR)/train/train.cu

MATRIX_TEST_FILE = $(TEST_DIR)/test_matrix.cpp
MODEL_TEST_FILE = $(TEST_DIR)/test_model.cpp

MATRIX_OUTPUT = $(BUILD_DIR)/test_matrix
MODEL_OUTPUT = $(BUILD_DIR)/test_model
TRAINING_OUTPUT = $(BUILD_DIR)/train

all: $(MATRIX_OUTPUT) $(DATA_OUTPUT) $(MODEL_OUTPUT) $(TRAINING_OUTPUT)

$(MATRIX_OUTPUT): $(MATRIX_SRC_FILES) $(MATRIX_TEST_FILE)
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(CFLAGS) -o $(MATRIX_OUTPUT) $(MATRIX_TEST_FILE) $(MATRIX_SRC_FILES)

$(MODEL_OUTPUT): $(MODEL_SRC_FILES) $(MODEL_TEST_FILE) $(MATRIX_SRC_FILES)
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(CFLAGS) -o $(MODEL_OUTPUT) $(MODEL_TEST_FILE) $(MODEL_SRC_FILES) $(MATRIX_SRC_FILES)

$(TRAINING_OUTPUT): $(TRAINING_SRC_FILES)
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(CFLAGS) -o $(TRAINING_OUTPUT) $(TRAINING_SRC_FILES) $(MATRIX_SRC_FILES) $(DATA_SRC_FILES) $(MODEL_SRC_FILES)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
