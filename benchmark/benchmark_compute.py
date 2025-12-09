import torch
import torch.nn as nn
import time
import numpy as np

# ==========================================
# 1. Define Model Architecture (Matches CUDA)
# ==========================================
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, output_size=10):
        super(MLP, self).__init__()
        # Matches main.cu: 784 -> 256 -> 10
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    def init_weights(self):
        """
        Match CUDA initialization: 
        - Xavier Uniform for weights
        - Zeros for biases
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2.0))
                nn.init.zeros_(m.bias)

# ==========================================
# 2. Benchmark Configuration
# ==========================================
BATCH_SIZE = 64
EPOCHS = 5
LR = 0.01
INPUT_DIM = 784
HIDDEN_DIM = 256
OUTPUT_DIM = 10
N_SAMPLES = 60000

def run_benchmark():
    # Check for CUDA
    if not torch.cuda.is_available():
        print("CRITICAL: CUDA is not available. This benchmark is meaningless on CPU.")
        return
    
    device = torch.device('cuda')
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # -------------------------------------------------
    # A. Setup Data (Synthetic to isolate compute speed)
    # -------------------------------------------------
    print(f"Generating synthetic data ({N_SAMPLES} samples)...")
    # Pre-load entire dataset to GPU memory (Matching main.cu behavior)
    # We use random data to skip disk I/O, ensuring we test TRAINING speed only.
    full_X = torch.randn(N_SAMPLES, INPUT_DIM, device=device)
    # Target (One-hot encoded to match typical CUDA C++ implementation logic)
    full_Y = torch.zeros(N_SAMPLES, OUTPUT_DIM, device=device)
    # Just set random class to 1 for validity
    indices = torch.randint(0, OUTPUT_DIM, (N_SAMPLES,), device=device)
    full_Y.scatter_(1, indices.unsqueeze(1), 1.0)

    # -------------------------------------------------
    # B. Setup Model
    # -------------------------------------------------
    model = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
    model.init_weights()
    
    # Use CrossEntropy. Note: If using one-hot targets, strict CrossEntropy might
    # require slightly different handling in older PyTorch, but works in newer.
    # For pure timing, MSE or CrossEntropy cost is negligible compared to MatMul.
    criterion = nn.CrossEntropyLoss()
    
    # -------------------------------------------------
    # C. Warmup
    # -------------------------------------------------
    print("Warming up GPU...")
    for _ in range(10):
        dummy_out = model(full_X[:BATCH_SIZE])
        loss = criterion(dummy_out, full_Y[:BATCH_SIZE])
        loss.backward()
        model.zero_grad()
    
    torch.cuda.synchronize()
    print("Warmup complete. Starting benchmark...")

    # -------------------------------------------------
    # D. Training Loop (Timed)
    # -------------------------------------------------
    num_batches = N_SAMPLES // BATCH_SIZE
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    
    for epoch in range(EPOCHS):
        for b in range(num_batches):
            # 1. Slice batch (Device-to-Device copy, matches CUDA implementation)
            # Note: Slicing in PyTorch is usually zero-copy (view), 
            # while cudaMemcpyDeviceToDevice is an actual copy. 
            # To be strictly fair against your CUDA memcpy, we can clone().
            start_idx = b * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
            
            batch_X = full_X[start_idx:end_idx] # .clone() if you want to force copy cost
            batch_Y = full_Y[start_idx:end_idx]
            
            # 2. Forward
            outputs = model(batch_X)
            
            # 3. Loss
            loss = criterion(outputs, batch_Y)
            
            # 4. Zero Grad
            model.zero_grad(set_to_none=True) # Slight optimization
            
            # 5. Backward
            loss.backward()
            
            # 6. Manual Update (Matches your CUDA C++ logic)
            with torch.no_grad():
                for param in model.parameters():
                    param -= LR * param.grad
            
            # CRITICAL: NO accuracy calculation here!
            # CRITICAL: NO printing/logging here!

    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_time_ms = start_event.elapsed_time(end_event)
    seconds = elapsed_time_ms / 1000.0
    
    print("-" * 40)
    print(f"Results for {EPOCHS} Epochs:")
    print(f"Total Training Time: {seconds:.4f} seconds")
    print(f"Avg Time per Epoch:  {seconds/EPOCHS:.4f} seconds")
    print("-" * 40)

if __name__ == "__main__":
    run_benchmark()