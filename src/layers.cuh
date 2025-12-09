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

        // UnFused version:
        // matrixMultiply(input, W, output);
        // addBias(output, b);
        
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

// -----------------------------------------------------------
// Linear Layer with Fused ReLU
// -----------------------------------------------------------
class LinearReLU : public Layer {
public:
    Matrix W, b;
    Matrix dW, db;
    Matrix input_cache;
    Matrix output;
    Matrix d_input;
    Matrix dZ_temp; // Temporary storage for dZ (gradient before ReLU)

    LinearReLU(int input_features, int output_features) {
        W.allocate(input_features, output_features);
        b.allocate(1, output_features);
    }

    ~LinearReLU() {
        W.free(); b.free();
        dW.free(); db.free();
        output.free();
        d_input.free();
        dZ_temp.free();
    }

    Matrix forward(const Matrix& input) override {
        input_cache = input; 

        if (!output.allocated || output.rows != input.rows || output.cols != W.cols) {
            output.allocate(input.rows, W.cols);
        }
        
        // Fused MatMul + Bias + ReLU
        matrixMultiplyWithBiasAndReLU(input, W, b, output);
        
        return output;
    }

    Matrix backward(const Matrix& d_output, float learning_rate) override {
        if (!d_input.allocated || d_input.rows != input_cache.rows || d_input.cols != input_cache.cols) 
            d_input.allocate(input_cache.rows, input_cache.cols);
        
        if (!dZ_temp.allocated || dZ_temp.rows != d_output.rows || dZ_temp.cols != d_output.cols)
            dZ_temp.allocate(d_output.rows, d_output.cols);

        if (!dW.allocated) dW.allocate(W.rows, W.cols);
        if (!db.allocated) db.allocate(b.rows, b.cols);

        // 1. ReLU Backward: dZ = d_output * (output > 0)
        // We use 'output' (which is A) as the proxy for Z. A > 0 implies Z > 0.
        reluBackward(d_output, output, dZ_temp);

        // 2. Linear Backward using dZ_temp
        matrixMultiplyTransposeA(input_cache, dZ_temp, dW);
        computeBiasGradient(dZ_temp, db);
        matrixMultiplyTransposeB(dZ_temp, W, d_input);

        updateWeights(W, dW, learning_rate);
        updateWeights(b, db, learning_rate);

        return d_input;
    }
};