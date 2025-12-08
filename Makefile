# Path to the virtual environment python
PYTHON := $(CURDIR)/.venv/bin/python
#
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
#
SRCS = $(SRC_DIR)/main.cu $(SRC_DIR)/kernels.cu
OBJS = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(SRCS))
TARGET = $(BIN_DIR)/mlp_test
#
all: $(TARGET)
#
$(TARGET): $(OBJS)
        @mkdir -p $(BIN_DIR)
        $(NVCC) $(NVCC_FLAGS) -o $@ $^
#
        $(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
        @mkdir -p $(OBJ_DIR)
        $(NVCC) $(NVCC_FLAGS) -c $< -o $@
#
xor: $(BIN_DIR)/mlp_xor_test
        @echo "Building MLP XOR Test..."
        ./$(BIN_DIR)/mlp_xor_test
#
        $(BIN_DIR)/mlp_xor_test: $(OBJ_DIR)/main_xor.o $(OBJ_DIR)/kernels.o
        @mkdir -p $(BIN_DIR)
        $(NVCC) $(NVCC_FLAGS) -o $@ $^
#
# Benchmark targets
benchmark-mnist: $(TARGET)
        @echo "Running MNIST benchmark..."
        cd benchmark && $(PYTHON) benchmark_mnist.py
benchmark-xor: $(BIN_DIR)/mlp_xor_test
        @echo "Running XOR benchmark..."
        cd benchmark && $(PYTHON) benchmark_xor.py
#
benchmark-all: $(TARGET) $(BIN_DIR)/mlp_xor_test
        @echo "Running all benchmarks..."
        cd benchmark && $(PYTHON) benchmark_mnist.py
        cd benchmark && $(PYTHON) benchmark_xor.py
        cd benchmark && $(PYTHON) visualize.py

clean:
		rm -rf $(OBJ_DIR) $(BIN_DIR)