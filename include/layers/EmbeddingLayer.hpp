#pragma once
#include "core/Tensor.hpp"
#include "core/Operation.hpp"
#include <vector>
#include <random> // Para inicialización de los embeddings
using namespace std;

class EmbeddingLayer : public Operation
{
private:
    Tensor embedding_weights_;   // Matriz de pesos de los embeddings (vocabulary_size x embedding_dimension)
    size_t vocabulary_size_;     // Tamaño del vocabulario
    size_t embedding_dimension_; // Dimensión de los embeddings

public:
    EmbeddingLayer(size_t vocabularySize, size_t embeddingDimension)
        : vocabulary_size_(vocabularySize), embedding_dimension_(embeddingDimension),
          embedding_weights_({vocabularySize, embeddingDimension})
    {
        // Inicialización de los pesos de los embeddings (puedes usar diferentes estrategias)
        std::random_device rd;
        std::mt19937 gen(rd());
        float stddev = 1.0f / std::sqrt(embeddingDimension);
        std::normal_distribution<float> distribution(0.0f, stddev);

        std::vector<float> weights_data(vocabularySize * embeddingDimension);
        for (float &w : weights_data)
        {
            w = distribution(gen);
        }
        embedding_weights_ = Tensor({vocabularySize, embeddingDimension}, weights_data);
    }
    ~EmbeddingLayer() override = default;

    Tensor forward(const std::vector<Tensor> &inputs)
    {
        if (inputs.size() != 1)
        {
            throw std::invalid_argument("EmbeddingLayer takes exactly one input tensor (indices).");
        }
        const Tensor &input_indices = inputs[0];
        const std::vector<size_t> &indices_shape = input_indices.getShape();
        std::vector<float> output_data;
        std::vector<size_t> output_shape = indices_shape; // La forma base es la misma que los índices
        output_shape.push_back(embedding_dimension_);     // Añadimos la dimensión del embedding

        const std::vector<float> &indices_data_float = input_indices.getData();
        std::vector<size_t> indices_data(indices_data_float.begin(), indices_data_float.end()); // Convertir a size_t

        for (size_t index : indices_data)
        {
            if (index >= vocabulary_size_)
            {
                throw std::out_of_range("Input index is out of vocabulary range.");
            }
            // Extraer el embedding correspondiente de embedding_weights_
            std::vector<size_t> embedding_index = {index};
            for (size_t i = 0; i < embedding_dimension_; ++i)
            {
                embedding_index.push_back(i);
                output_data.push_back(embedding_weights_.at(embedding_index));
                embedding_index.pop_back();
            }
        }
        return Tensor(output_shape, output_data);
    }

    Tensor backward(const Tensor &outputGradient, const std::vector<Tensor> &inputs)
    {
        if (inputs.size() != 1)
        {
            throw std::invalid_argument("EmbeddingLayer takes exactly one input tensor (indices) for backward pass.");
        }
        const Tensor &input_indices = inputs[0];
        const std::vector<float> &indices_data_float = input_indices.getData();
        std::vector<size_t> indices_data(indices_data_float.begin(), indices_data_float.end()); // Convertir a size_t

        Tensor weight_gradients({vocabulary_size_, embedding_dimension_});
        const std::vector<float> &output_grad_data = outputGradient.getData();
        const std::vector<size_t> &output_grad_shape = outputGradient.getShape();

        size_t embedding_dim = output_grad_shape.back();
        size_t num_elements = output_grad_data.size() / embedding_dim;

        for (size_t i = 0; i < num_elements; ++i)
        {
            size_t index = indices_data[i];
            if (index >= vocabulary_size_)
                continue; // Ignore gradients for out-of-vocabulary tokens

            for (size_t j = 0; j < embedding_dim; ++j)
            {
                std::vector<size_t> weight_indices = {index, j};
                std::vector<size_t> grad_indices(output_grad_shape.begin(), output_grad_shape.end() - 1);
                grad_indices.push_back(j);

                float grad_value = 1.0f; // Inicializar con 1 para el primer elemento
                size_t grad_offset = 0;
                size_t current_stride = 1;
                for (size_t k = grad_indices.size(); k > 0; --k)
                {
                    grad_offset += grad_indices[k - 1] * current_stride;
                    if (k > 1)
                    {
                        current_stride *= output_grad_shape[k - 2];
                    }
                }
                grad_value = output_grad_data[i * embedding_dim + j];
                weight_gradients.at(weight_indices) += grad_value;
            }
        }

        // The backward pass for the input indices doesn't typically produce a meaningful gradient
        // in the same way as for weights. We can return a zero tensor of the input shape.
        std::vector<float> zero_gradient_data(input_indices.getData().size(), 0.0f);
        return Tensor(input_indices.getShape(), zero_gradient_data);
    }
};