NVCC = nvcc

CFLAGS = -I./include

SRC_DIR = src
INC_DIR = include
TEST_DIR = tests
BUILD_DIR = build

SRC_FILES = $(SRC_DIR)/vector.cu $(SRC_DIR)/vector_kernels.cu
TEST_FILES = $(TEST_DIR)/test_vector.cpp

OUTPUT = $(BUILD_DIR)/run_tests

build: $(OUTPUT)

$(OUTPUT): $(SRC_FILES) $(TEST_FILES)
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(CFLAGS) -o $(OUTPUT) $(TEST_FILES) $(SRC_FILES)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all build clean
