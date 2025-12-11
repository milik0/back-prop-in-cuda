# Path to the virtual environment python
PYTHON := $(CURDIR)/.venv/bin/python
NVCC := nvcc
NVCC_FLAGS := -arch=sm_61 -O3 -std=c++17 -ccbin g++-10


SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

SRCS = $(SRC_DIR)/main.cu $(SRC_DIR)/kernels.cu $(SRC_DIR)/kernels_naive.cu $(SRC_DIR)/kernels_shared.cu $(SRC_DIR)/kernels_fused.cu $(SRC_DIR)/kernels_warp.cu
KERNEL_SRCS = $(SRC_DIR)/kernels.cu $(SRC_DIR)/kernels_naive.cu $(SRC_DIR)/kernels_shared.cu $(SRC_DIR)/kernels_fused.cu $(SRC_DIR)/kernels_warp.cu
KERNEL_OBJS = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(KERNEL_SRCS))
OBJS = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(SRCS))
TARGET = $(BIN_DIR)/mlp_test

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(BIN_DIR)/mlp_xor_test: $(OBJ_DIR)/main_xor.o $(OBJ_DIR)/kernels.o
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

$(BIN_DIR)/mlp_mnist_test: $(OBJ_DIR)/main.o $(KERNEL_OBJS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

$(BIN_DIR)/mlp_mnist_train_test: $(OBJ_DIR)/main_mnist.o $(KERNEL_OBJS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

$(BIN_DIR)/mlp_fashion_mnist_test: $(OBJ_DIR)/main_fashion_mnist.o $(OBJ_DIR)/kernels.o
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^
$(BIN_DIR)/mlp_breast_cancer_test: $(OBJ_DIR)/main_breast_cancer.o $(OBJ_DIR)/kernels.o
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

xor: $(BIN_DIR)/mlp_xor_test
	@echo "Building MLP XOR Test..."
	./$(BIN_DIR)/mlp_xor_test

mnist: $(BIN_DIR)/mlp_mnist_test
	@echo "Building MLP XOR Test..."
	./$(BIN_DIR)/mlp_mnist_test

fashion-mnist: $(BIN_DIR)/mlp_fashion_mnist_test
	@echo "Building MLP Fashion-MNIST Test..."
	./$(BIN_DIR)/mlp_fashion_mnist_test

breast-cancer: $(BIN_DIR)/mlp_breast_cancer_test
	@echo "Building MLP Breast Cancer Test..."
	./$(BIN_DIR)/mlp_breast_cancer_test


# Benchmark targets
benchmark-mnist: $(BIN_DIR)/mlp_mnist_train_test
	@echo "Running MNIST benchmark..."
	cd benchmark && $(PYTHON) benchmark_mnist.py
benchmark-xor: $(BIN_DIR)/mlp_xor_test
	@echo "Running XOR benchmark..."
	cd benchmark && $(PYTHON) benchmark_xor.py

benchmark-fashion-mnist: $(TARGET)
	@echo "Running Fashion-MNIST benchmark..."
	cd benchmark && $(PYTHON) benchmark_fashion_mnist.py

benchmark-breast-cancer: $(TARGET)
	@echo "Running Breast Cancer benchmark..."
	cd benchmark && $(PYTHON) benchmark_breast_cancer.py

benchmark-compute:
	@echo "Running layer comparison benchmark..."
	cd benchmark && $(PYTHON) benchmark_compute.py


benchmark-all: $(TARGET) $(BIN_DIR)/mlp_xor_test
	@echo "Running all benchmarks..."
	cd benchmark && $(PYTHON) benchmark_mnist.py
	cd benchmark && $(PYTHON) benchmark_xor.py
	cd benchmark && $(PYTHON) visualize.py

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
