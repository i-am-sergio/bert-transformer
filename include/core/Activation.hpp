#pragma once
#include "Tensor.hpp"
#include "Operation.hpp"
#include <vector>

class Activation : public Operation
{
public:
    virtual ~Activation() = default;
};

class ReLU : public Activation
{
public:
    Tensor forward(const std::vector<Tensor> &inputs)
    {
        if (inputs.size() != 1)
        {
            throw std::invalid_argument("ReLU takes exactly one input tensor.");
        }
        Tensor input = inputs[0];
        std::vector<float> output_data(input.getData().begin(), input.getData().end());
        for (float &val : output_data)
            val = std::max(0.0f, val);
        return Tensor(input.getShape(), output_data);
    }
    Tensor backward(const Tensor &outputGradient, const std::vector<Tensor> &inputs) override
    {
        if (inputs.size() != 1)
        {
            throw std::invalid_argument("ReLU takes exactly one input tensor for backward pass.");
        }
        const Tensor &input = inputs[0];
        const std::vector<float> &input_data = input.getData();
        const std::vector<float> &grad_data = outputGradient.getData();
        std::vector<float> input_gradient_data(input_data.size());
        for (size_t i = 0; i < input_data.size(); ++i)
        {
            if (input_data[i] > 0.0f)
                input_gradient_data[i] = grad_data[i];
            else
                input_gradient_data[i] = 0.0f;
        }
        return Tensor(input.getShape(), input_gradient_data);
    }
};