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

xor: SRC = $(SRC_DIR)/main_xor.cu $(SRC_DIR)/kernels.cu
xor: OBJS = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(SRC))
xor: TARGET = $(BIN_DIR)/mlp_xor_test
xor: @echo "Building MLP XOR Test..."
	 @echo ${TARGET}
xor: ./$(TARGET)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

run: $(TARGET)
	./$(TARGET)
