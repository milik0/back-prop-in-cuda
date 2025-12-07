#pragma once
#include "utils.cuh"
#include "kernels.cuh"

// Abstract Base Class
class Layer {
public:
    virtual ~Layer() {}
    
    // Returns the output matrix
    virtual Matrix forward(const Matrix& input) = 0;
    
    // Returns the gradient with respect to the input (d_input)
    // Receives the gradient of the loss with respect to the output (d_output)
    virtual Matrix backward(const Matrix& d_output, float learning_rate) = 0;
};

// -----------------------------------------------------------
// Linear (Dense) Layer: Output = Input * W + b
// -----------------------------------------------------------
class Linear : public Layer {
public:
    Matrix W, b;      // Parameters
    Matrix dW, db;    // Gradients
    Matrix input_cache; // Cache for backward pass
    Matrix output;    // Storage for forward output
    Matrix d_input;   // Storage for backward output

    Linear(int input_features, int output_features) {
        W.allocate(input_features, output_features);
        b.allocate(1, output_features);
        
        // In a real app, initialize W with random values here!
        // For now, we assume user initializes W/b manually or via a helper.
    }

    ~Linear() {
        W.free(); b.free();
        dW.free(); db.free();
        output.free();
        d_input.free();
    }

    Matrix forward(const Matrix& input) override {
        // Cache input for backward pass
        // Deep copy is safer if the previous layer overwrites its output, 
        // but for sequential MLPs, shallow copy or reference is often enough.
        // Here we assume 'input' stays valid until backward.
        input_cache = input; 

        // Prepare Output
        if (!output.allocated) output.allocate(input.rows, W.cols);
        
        // 1. Z = X * W
        matrixMultiply(input, W, output);
        
        // 2. Z = Z + b
        addBias(output, b);
        
        return output;
    }

    Matrix backward(const Matrix& d_output, float learning_rate) override {
        // Dimensions:
        // Input (X): M x K
        // Weights (W): K x N
        // d_output (dZ): M x N
        
        if (!d_input.allocated) d_input.allocate(input_cache.rows, input_cache.cols);
        if (!dW.allocated) dW.allocate(W.rows, W.cols);
        if (!db.allocated) db.allocate(b.rows, b.cols);

        // 1. Compute dW = X^T * dZ
        // You need to implement matmul_transpose_A in kernels.cu
        matrixMultiplyTransposeA(input_cache, d_output, dW);

        // 2. Compute db = sum(dZ, axis=0)
        // You need to implement sum_columns in kernels.cu
        computeBiasGradient(d_output, db);

        // 3. Compute d_input = dZ * W^T
        // You need to implement matmul_transpose_B in kernels.cu
        matrixMultiplyTransposeB(d_output, W, d_input);

        // 4. Update Weights (Gradient Descent)
        // W = W - lr * dW
        // b = b - lr * db
        updateWeights(W, dW, learning_rate);
        updateWeights(b, db, learning_rate);

        return d_input;
    }
};

// -----------------------------------------------------------
// ReLU Layer: Activation
// -----------------------------------------------------------
class ReLU : public Layer {
public:
    Matrix input_cache;
    Matrix output;
    Matrix d_input;

    ~ReLU() {
        output.free();
        d_input.free();
    }

    Matrix forward(const Matrix& input) override {
        input_cache = input;
        if (!output.allocated) output.allocate(input.rows, input.cols);

        reluActivation(input, output);
        return output;
    }

    Matrix backward(const Matrix& d_output, float learning_rate) override {
        if (!d_input.allocated) d_input.allocate(d_output.rows, d_output.cols);

        // d_input = d_output * (input > 0 ? 1 : 0)
        reluBackward(d_output, input_cache, d_input);
        
        return d_input;
    }
};

class SoftmaxCrossEntropy : public Layer {
public:
    Matrix output;
    Matrix d_input;

    ~SoftmaxCrossEntropy() { output.free(); d_input.free(); }

    Matrix forward(const Matrix& input) override {
        if (!output.allocated) output.allocate(input.rows, input.cols);
        softmaxActivation(input, output);
        return output;
    }

    Matrix backward(const Matrix& target_labels, float learning_rate) override {
        // For Softmax + CrossEntropy, the gradient passed to the previous layer
        // is simply: (Prediction - Target)
        // Here, 'target_labels' acts as the ground truth Y.
        
        if (!d_input.allocated) d_input.allocate(target_labels.rows, target_labels.cols);
        
        // We reuse the subtract kernel: d_input = output - target_labels
        // You might need to expose a generic "subtract" kernel or use the MSE one
        computeMSEGradient(output, target_labels, d_input); 
        
        return d_input;
    }
};