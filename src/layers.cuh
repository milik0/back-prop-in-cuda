#pragma once
#include "utils.cuh"
#include "kernels.cuh"

// Abstract Base Class
class Layer {
public:
    virtual ~Layer() {}
    virtual Matrix forward(const Matrix& input) = 0;
    virtual Matrix backward(const Matrix& d_output, float learning_rate) = 0;
};

// -----------------------------------------------------------
// Linear Layer
// -----------------------------------------------------------
class Linear : public Layer {
public:
    Matrix W, b;
    Matrix dW, db;
    Matrix input_cache;
    Matrix output;
    Matrix d_input;

    Linear(int input_features, int output_features) {
        W.allocate(input_features, output_features);
        b.allocate(1, output_features);
    }

    ~Linear() {
        W.free(); b.free();
        dW.free(); db.free();
        // Do not free input_cache (we don't own it)
        output.free();
        d_input.free();
    }

    Matrix forward(const Matrix& input) override {
        input_cache = input; 

        // FIX: Check if batch size changed, re-allocate if necessary
        if (!output.allocated || output.rows != input.rows || output.cols != W.cols) {
            output.allocate(input.rows, W.cols);
        }
        
        // Fused MatMul + Bias
        matrixMultiplyWithBias(input, W, b, output);
        
        return output;
    }

    Matrix backward(const Matrix& d_output, float learning_rate) override {
        if (!d_input.allocated || d_input.rows != input_cache.rows || d_input.cols != input_cache.cols) 
            d_input.allocate(input_cache.rows, input_cache.cols);
        
        if (!dW.allocated) dW.allocate(W.rows, W.cols);
        if (!db.allocated) db.allocate(b.rows, b.cols);

        matrixMultiplyTransposeA(input_cache, d_output, dW);
        computeBiasGradient(d_output, db);
        matrixMultiplyTransposeB(d_output, W, d_input);

        updateWeights(W, dW, learning_rate);
        updateWeights(b, db, learning_rate);

        return d_input;
    }
};

// -----------------------------------------------------------
// ReLU Layer
// -----------------------------------------------------------
class ReLU : public Layer {
public:
    Matrix input_cache;
    Matrix output;
    Matrix d_input;

    ~ReLU() { output.free(); d_input.free(); }

    Matrix forward(const Matrix& input) override {
        input_cache = input;
        
        // FIX: Re-allocate if dimensions change
        if (!output.allocated || output.rows != input.rows || output.cols != input.cols) {
            output.allocate(input.rows, input.cols);
        }

        reluActivation(input, output);
        return output;
    }

    Matrix backward(const Matrix& d_output, float learning_rate) override {
        if (!d_input.allocated || d_input.rows != d_output.rows || d_input.cols != d_output.cols) 
            d_input.allocate(d_output.rows, d_output.cols);

        reluBackward(d_output, input_cache, d_input);
        return d_input;
    }
};

// -----------------------------------------------------------
// Softmax Cross Entropy Layer
// -----------------------------------------------------------
class SoftmaxCrossEntropy : public Layer {
public:
    Matrix output;
    Matrix d_input;

    ~SoftmaxCrossEntropy() { output.free(); d_input.free(); }

    Matrix forward(const Matrix& input) override {
        // FIX: Re-allocate if dimensions change
        if (!output.allocated || output.rows != input.rows || output.cols != input.cols) {
            output.allocate(input.rows, input.cols);
        }
        
        softmaxActivation(input, output);
        return output;
    }

    Matrix backward(const Matrix& target_labels, float learning_rate) override {
        if (!d_input.allocated || d_input.rows != target_labels.rows || d_input.cols != target_labels.cols) 
            d_input.allocate(target_labels.rows, target_labels.cols);
        
        // For Softmax+CE, gradient is (Pred - Target)
        // We reuse the generic subtract/MSE gradient kernel
        computeMSEGradient(output, target_labels, d_input); 
        
        return d_input;
    }
};