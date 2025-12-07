NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_75 --std=c++14

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

SRCS = $(SRC_DIR)/main.cu $(SRC_DIR)/kernels.cu
OBJS = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(SRCS))
TARGET = $(BIN_DIR)/mlp_test

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

xor: $(BIN_DIR)/mlp_xor_test
	@echo "Building MLP XOR Test..."
	./$(BIN_DIR)/mlp_xor_test

$(BIN_DIR)/mlp_xor_test: $(OBJ_DIR)/main_xor.o $(OBJ_DIR)/kernels.o
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Benchmark targets
benchmark-mnist: $(TARGET)
	@echo "Running MNIST benchmark..."
	cd benchmark && python3 benchmark_mnist.py

benchmark-xor: $(BIN_DIR)/mlp_xor_test
	@echo "Running XOR benchmark..."
	cd benchmark && python3 benchmark_xor.py

benchmark-all: $(TARGET) $(BIN_DIR)/mlp_xor_test
	@echo "Running all benchmarks..."
	cd benchmark && python3 benchmark_mnist.py
	cd benchmark && python3 benchmark_xor.py
	cd benchmark && python3 visualize.py

benchmark-visualize:
	@echo "Generating visualizations..."
	cd benchmark && python3 visualize.py

benchmark-setup:
	@echo "Installing Python dependencies for benchmarks..."
	pip3 install -r benchmark/requirements.txt

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

run: $(TARGET)
	./$(TARGET)
