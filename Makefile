NVCC = nvcc

CFLAGS = -I./include -I./src -I./src/matrix -std=c++20
NVCC_FLAGS = -arch=sm_90 -gencode=arch=compute_90,code=sm_90
ifeq ($(DEBUG), 1)
    CFLAGS += -g
    NVCC_FLAGS += -G
endif

SRC_DIR = src
TEST_DIR = tests
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj

ALLOCATOR_SRC_FILES = $(SRC_DIR)/allocator/allocator.cu
DATA_SRC_FILES = $(SRC_DIR)/dataloader/read_mnist.cpp
MODEL_SRC_FILES = $(SRC_DIR)/model/model.cpp
TRAINING_SRC_FILES = $(SRC_DIR)/train/config_reader.cpp $(SRC_DIR)/train/train.cu
UTILS_SRC_FILES = $(SRC_DIR)/utils/cuda_utils.cu

ALLOCATOR_TEST_FILE = $(TEST_DIR)/test_allocator.cu
MATRIX_TEST_FILE = $(TEST_DIR)/test_matrix.cu
MODEL_TEST_FILE = $(TEST_DIR)/test_model.cpp

ALLOCATOR_OUTPUT = $(BUILD_DIR)/test_allocator
MATRIX_OUTPUT = $(BUILD_DIR)/test_matrix
MODEL_OUTPUT = $(BUILD_DIR)/test_model
TRAINING_OUTPUT = $(BUILD_DIR)/train

ALLOCATOR_OBJ_FILES = $(ALLOCATOR_SRC_FILES:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
DATA_OBJ_FILES = $(DATA_SRC_FILES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
MODEL_OBJ_FILES = $(MODEL_SRC_FILES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
TRAINING_OBJ_FILES = $(TRAINING_SRC_FILES:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
UTILS_OBJ_FILES = $(UTILS_SRC_FILES:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)

all: $(ALLOCATOR_OUTPUT) $(MATRIX_OUTPUT) # $(MODEL_OUTPUT) $(TRAINING_OUTPUT)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	mkdir -p $(dir $@)
	$(NVCC) $(CFLAGS) $(NVCC_FLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	mkdir -p $(dir $@)
	$(NVCC) $(CFLAGS) $(NVCC_FLAGS) -x cu -c $< -o $@

$(ALLOCATOR_OUTPUT): $(ALLOCATOR_TEST_FILE) $(ALLOCATOR_OBJ_FILES) $(UTILS_OBJ_FILES)
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(CFLAGS) $(NVCC_FLAGS) -o $@ $^

$(MATRIX_OUTPUT): $(MATRIX_TEST_FILE) $(ALLOCATOR_OBJ_FILES) $(UTILS_OBJ_FILES)
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(CFLAGS) $(NVCC_FLAGS) -o $@ $^

$(MODEL_OUTPUT): $(MODEL_TEST_FILE) $(MODEL_OBJ_FILES) $(MATRIX_OBJ_FILES) $(ALLOCATOR_OBJ_FILES) $(UTILS_OBJ_FILES)
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(CFLAGS) $(NVCC_FLAGS) -o $@ $^

$(TRAINING_OUTPUT): $(TRAINING_OBJ_FILES) $(MODEL_OBJ_FILES) $(MATRIX_OBJ_FILES) $(DATA_OBJ_FILES) $(ALLOCATOR_OBJ_FILES) $(UTILS_OBJ_FILES)
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(CFLAGS) $(NVCC_FLAGS) -o $@ $^

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
