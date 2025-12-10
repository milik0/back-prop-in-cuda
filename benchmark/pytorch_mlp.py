"""
PyTorch MLP implementation matching the CUDA version
Architecture: 784 -> 256 -> 10 for MNIST
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class MLP(nn.Module):
    """
    Multi-Layer Perceptron matching CUDA implementation
    """
    def __init__(self, input_size=784, hidden_size=256, output_size=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    def init_xavier(self):
        """Initialize weights using Xavier initialization to match CUDA"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2.0))
                nn.init.zeros_(m.bias)


class XORMLP(nn.Module):
    """
    Simple MLP for XOR problem matching CUDA implementation
    Architecture: 2 -> 8 -> 1
    """
    def __init__(self):
        super(XORMLP, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class FashionMLP(nn.Module):
    """
    MLP for Fashion-MNIST matching CUDA implementation
    Architecture: 784 -> 128 -> 10
    """
    def __init__(self):
        super(FashionMLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    def init_xavier(self):
        """Initialize weights using Xavier initialization to match CUDA"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2.0))
                nn.init.zeros_(m.bias)


class BreastCancerMLP(nn.Module):
    """
    MLP for Breast Cancer Wisconsin matching CUDA implementation
    Architecture: 9 -> 16 -> 8 -> 1
    """
    def __init__(self, input_dim=9):
        super(BreastCancerMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(8, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
    
    def init_xavier(self):
        """Initialize weights using Xavier initialization to match CUDA"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2.0))
                nn.init.zeros_(m.bias)


def load_mnist_pytorch(image_path, label_path, num_samples):
    """
    Load MNIST data similar to CUDA implementation
    """
    import struct
    
    # Load images
    with open(image_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8, count=num_samples * rows * cols)
        images = images.reshape(num_samples, rows * cols).astype(np.float32) / 255.0
    
    # Load labels (one-hot encoded)
    with open(label_path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8, count=num_samples)
        
    # One-hot encode labels
    labels_onehot = np.zeros((num_samples, 10), dtype=np.float32)
    labels_onehot[np.arange(num_samples), labels] = 1.0
    
    return torch.from_numpy(images), torch.from_numpy(labels_onehot)


def calculate_accuracy(predictions, targets):
    """Calculate accuracy matching CUDA implementation"""
    pred_labels = torch.argmax(predictions, dim=1)
    target_labels = torch.argmax(targets, dim=1)
    correct = (pred_labels == target_labels).sum().item()
    return correct / predictions.size(0)


def train_mnist(model, device, data_path, epochs=5, batch_size=64, lr=0.01, seed=1337):
    """
    Train MLP on MNIST matching CUDA implementation
    Returns: training history and timing information
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load data
    train_images = f"{data_path}/train-images-idx3-ubyte"
    train_labels = f"{data_path}/train-labels-idx1-ubyte"
    test_images = f"{data_path}/t10k-images-idx3-ubyte"
    test_labels = f"{data_path}/t10k-labels-idx1-ubyte"
    
    X_train, Y_train = load_mnist_pytorch(train_images, train_labels, 60000)
    X_test, Y_test = load_mnist_pytorch(test_images, test_labels, 10000)
    
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_test, Y_test = X_test.to(device), Y_test.to(device)
    
    # Initialize model
    model = model.to(device)
    model.init_xavier()
    
    criterion = nn.CrossEntropyLoss()
    # No optimizer to match CUDA's manual weight updates with fixed LR
    
    num_batches = 60000 // batch_size
    history = {'epoch': [], 'train_acc': [], 'test_acc': [], 'time': []}
    
    import time
    total_time = 0
    
    for epoch in range(epochs):
        epoch_start = time.time()
        total_acc = 0.0
        
        for b in range(num_batches):
            # Get batch
            start_idx = b * batch_size
            end_idx = start_idx + batch_size
            batch_X = X_train[start_idx:end_idx]
            batch_Y = Y_train[start_idx:end_idx]
            
            # Forward pass
            outputs = model(batch_X)
            
            # Loss and backward (manual gradient descent to match CUDA)
            loss = criterion(outputs, batch_Y)
            model.zero_grad()
            loss.backward()
            
            # Manual SGD update to match CUDA
            with torch.no_grad():
                for param in model.parameters():
                    param -= lr * param.grad
            
            # Calculate accuracy
            with torch.no_grad():
                predictions = torch.softmax(outputs, dim=1)
                acc = calculate_accuracy(predictions, batch_Y)
                total_acc += acc
        
        epoch_time = time.time() - epoch_start
        total_time += epoch_time
        
        avg_train_acc = (total_acc / num_batches) * 100.0
        
        # Test accuracy
        with torch.no_grad():
            test_outputs = model(X_test)
            test_predictions = torch.softmax(test_outputs, dim=1)
            test_acc = calculate_accuracy(test_predictions, Y_test) * 100.0
        
        history['epoch'].append(epoch)
        history['train_acc'].append(avg_train_acc)
        history['test_acc'].append(test_acc)
        history['time'].append(epoch_time)
        
        print(f"Epoch {epoch} | Train Acc: {avg_train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Time: {epoch_time:.3f}s")
    
    return history, total_time

def train_xor(model, device, epochs=10000, lr=0.01, seed=1337):
    """
    Train MLP on XOR problem matching CUDA implementation
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # XOR dataset
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)
    
    X, Y = X.to(device), Y.to(device)
    
    model = model.to(device)
    
    # Random initialization to match CUDA
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, -0.5, 0.5)
            nn.init.uniform_(m.bias, -0.5, 0.5)
    
    criterion = nn.MSELoss()
    
    history = {'epoch': [], 'loss': []}
    
    import time
    
    # Warmup to avoid measuring compilation time (matching CUDA implementation)
    for _ in range(10):
        outputs = model(X)
        loss = criterion(outputs, Y)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                param -= lr * param.grad
    
    # Synchronize before starting timer
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Forward
        outputs = model(X)
        loss = criterion(outputs, Y)
        
        # Backward and update (manual SGD)
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            for param in model.parameters():
                param -= lr * param.grad
        
        # Store loss WITHOUT calling .item() to avoid sync
        if epoch % 1000 == 0:
            history['epoch'].append(epoch)
            # Clone to avoid holding reference to computation graph
            history['loss'].append(loss.detach().clone())
    
    # Synchronize before stopping timer
    if device == 'cuda':
        torch.cuda.synchronize()
    
    total_time = time.time() - start_time
    
    # Convert losses to CPU after training
    history['loss'] = [l.item() for l in history['loss']]
    
    # Print logged losses
    for epoch, loss_val in zip(history['epoch'], history['loss']):
        print(f"Epoch {epoch:5d} | Loss: {loss_val:.6f}")
    
    # Final predictions
    with torch.no_grad():
        final_outputs = model(X)
        print("\nFinal Predictions:")
        for i in range(4):
            print(f"Input: {X[i].cpu().numpy()} -> Output: {final_outputs[i].item():.4f} (Target: {Y[i].item()})")
    
    return history, total_time


def train_fashion_mnist(model, device, data_path, epochs=10, batch_size=64, lr=0.01, seed=1337):
    """
    Train MLP on Fashion-MNIST matching CUDA implementation
    Returns: training history and timing information
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load data (Fashion-MNIST uses same format as MNIST)
    train_images = f"{data_path}/train-images-idx3-ubyte"
    train_labels = f"{data_path}/train-labels-idx1-ubyte"
    test_images = f"{data_path}/t10k-images-idx3-ubyte"
    test_labels = f"{data_path}/t10k-labels-idx1-ubyte"
    
    X_train, Y_train = load_mnist_pytorch(train_images, train_labels, 60000)
    X_test, Y_test = load_mnist_pytorch(test_images, test_labels, 10000)
    
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_test, Y_test = X_test.to(device), Y_test.to(device)
    
    # Initialize model
    model = model.to(device)
    model.init_xavier()
    
    criterion = nn.CrossEntropyLoss()
    
    num_batches = 60000 // batch_size
    history = {'epoch': [], 'train_acc': [], 'test_acc': [], 'time': []}
    
    import time
    
    # Warmup
    for b in range(10):
        start_idx = b * batch_size
        end_idx = start_idx + batch_size
        batch_X = X_train[start_idx:end_idx]
        batch_Y = Y_train[start_idx:end_idx]
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                param -= lr * param.grad
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    total_time = 0
    
    for epoch in range(epochs):
        epoch_start = time.time()
        total_acc = 0.0
        
        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = start_idx + batch_size
            batch_X = X_train[start_idx:end_idx]
            batch_Y = Y_train[start_idx:end_idx]
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            model.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                for param in model.parameters():
                    param -= lr * param.grad
            
            with torch.no_grad():
                predictions = torch.softmax(outputs, dim=1)
                acc = calculate_accuracy(predictions, batch_Y)
                total_acc += acc
        
        epoch_time = time.time() - epoch_start
        total_time += epoch_time
        
        avg_train_acc = (total_acc / num_batches) * 100.0
        
        with torch.no_grad():
            test_outputs = model(X_test)
            test_predictions = torch.softmax(test_outputs, dim=1)
            test_acc = calculate_accuracy(test_predictions, Y_test) * 100.0
        
        history['epoch'].append(epoch)
        history['train_acc'].append(avg_train_acc)
        history['test_acc'].append(test_acc)
        history['time'].append(epoch_time)
        
        print(f"Epoch {epoch} | Train Acc: {avg_train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Time: {epoch_time:.3f}s")
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    return history, total_time


def load_breast_cancer_ubyte(image_path, label_path, num_samples):
    """
    Load Breast Cancer Wisconsin data from ubyte format
    """
    import struct
    
    # Load features (images file)
    with open(image_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        features = np.fromfile(f, dtype=np.uint8, count=num_samples * rows * cols)
        features = features.reshape(num_samples, rows * cols).astype(np.float32) / 255.0
    
    # Load labels
    with open(label_path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8, count=num_samples).astype(np.float32).reshape(-1, 1)
    
    return torch.from_numpy(features), torch.from_numpy(labels)


def train_breast_cancer(model, device, data_path, epochs=50, batch_size=32, lr=0.01, seed=1337):
    """
    Train MLP on Breast Cancer Wisconsin matching CUDA implementation
    Returns: training history and timing information
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load data from ubyte format
    image_path = f"{data_path}/bcw_original-images-ubyte"
    label_path = f"{data_path}/bcw_original-labels-ubyte"
    
    num_samples = 569  # Total samples in Breast Cancer Wisconsin dataset
    X, y = load_breast_cancer_ubyte(image_path, label_path, num_samples)
    
    # Split train/test (80/20)
    train_size = int(len(X) * 0.8)
    X_train = torch.from_numpy(X[:train_size])
    y_train = torch.from_numpy(y[:train_size])
    X_test = torch.from_numpy(X[train_size:])
    y_test = torch.from_numpy(y[train_size:])
    
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)
    
    print(f"Loaded Breast Cancer data: {len(X)} samples, {X.shape[1]} features")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Initialize model
    model = model.to(device)
    model.init_xavier()
    
    criterion = nn.MSELoss()
    
    num_batches = train_size // batch_size
    history = {'epoch': [], 'train_acc': [], 'test_acc': [], 'time': []}
    
    import time
    
    # Binary classification accuracy
    def binary_accuracy(preds, targets):
        pred_classes = (preds > 0.5).float()
        correct = (pred_classes == targets).sum().item()
        return correct / len(targets)
    
    # Warmup
    for b in range(min(10, num_batches)):
        start_idx = b * batch_size
        end_idx = start_idx + batch_size
        batch_X = X_train[start_idx:end_idx]
        batch_y = y_train[start_idx:end_idx]
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                param -= lr * param.grad
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        total_acc = 0.0
        
        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = start_idx + batch_size
            batch_X = X_train[start_idx:end_idx]
            batch_y = y_train[start_idx:end_idx]
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            model.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                for param in model.parameters():
                    param -= lr * param.grad
                
                acc = binary_accuracy(outputs, batch_y)
                total_acc += acc
        
        epoch_time = time.time() - epoch_start
        
        avg_train_acc = (total_acc / num_batches) * 100.0
        
        # Test accuracy
        with torch.no_grad():
            test_batches = len(X_test) // batch_size
            test_acc_sum = 0.0
            for b in range(test_batches):
                start_idx = b * batch_size
                end_idx = start_idx + batch_size
                batch_X = X_test[start_idx:end_idx]
                batch_y = y_test[start_idx:end_idx]
                outputs = model(batch_X)
                test_acc_sum += binary_accuracy(outputs, batch_y)
            test_acc = (test_acc_sum / test_batches) * 100.0
        
        history['epoch'].append(epoch)
        history['train_acc'].append(avg_train_acc)
        history['test_acc'].append(test_acc)
        history['time'].append(epoch_time)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Train Acc: {avg_train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Time: {epoch_time:.3f}s")
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    total_time = time.time() - start_time
    
    return history, total_time