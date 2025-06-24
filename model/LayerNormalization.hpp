#pragma once

#include "Tensor.hpp"
#include <cmath>

class LayerNormalization {
public:
    LayerNormalization(size_t embedding_dim) {
        gamma = Tensor({embedding_dim});  // Parámetro de escala
        beta = Tensor({embedding_dim});   // Parámetro de desplazamiento

        // Inicializar gamma y beta (normalmente se inicializan como 1 y 0 respectivamente)
        gamma.fill(1.0);
        beta.fill(0.0);
    }

    // Normalización de capa: Recibe la entrada y devuelve la salida normalizada
    Tensor forward(const Tensor &input) {
        size_t batch_size = input.shape[0];
        size_t sequence_length = input.shape[1];
        
        // Calcular la media y la desviación estándar de las activaciones
        Tensor mean = compute_mean(input);  // [batch_size, sequence_length]
        Tensor variance = compute_variance(input, mean);  // [batch_size, sequence_length]

        // Normalizar las activaciones
        Tensor normed = normalize(input, mean, variance);  // [batch_size, sequence_length, embedding_dim]

        // Aplicar gamma y beta
        Tensor output = normed * gamma;  // Escalar
        output = output + beta;  // Desplazar

        return output;
    }

private:
    Tensor gamma, beta;  // Parámetros de escala y desplazamiento

    // Calcular la media a lo largo de la última dimensión (embedding_dim)
    Tensor compute_mean(const Tensor &input) {
        Tensor mean(input.shape);
        for (size_t i = 0; i < input.shape[0]; ++i) {
            for (size_t j = 0; j < input.shape[1]; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < input.shape[2]; ++k) {
                    sum += input({i, j, k});
                }
                mean({i, j}) = sum / input.shape[2];
            }
        }
        return mean;
    }

    // Calcular la varianza a lo largo de la última dimensión (embedding_dim)
    Tensor compute_variance(const Tensor &input, const Tensor &mean) {
        Tensor variance(input.shape);
        for (size_t i = 0; i < input.shape[0]; ++i) {
            for (size_t j = 0; j < input.shape[1]; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < input.shape[2]; ++k) {
                    sum += std::pow(input({i, j, k}) - mean({i, j}), 2);
                }
                variance({i, j}) = sum / input.shape[2];
            }
        }
        return variance;
    }

    // Normalizar las activaciones
    Tensor normalize(const Tensor &input, const Tensor &mean, const Tensor &variance) {
        Tensor result(input.shape);
        for (size_t i = 0; i < input.shape[0]; ++i) {
            for (size_t j = 0; j < input.shape[1]; ++j) {
                for (size_t k = 0; k < input.shape[2]; ++k) {
                    float normalized = (input({i, j, k}) - mean({i, j})) / std::sqrt(variance({i, j}) + 1e-6);
                    result({i, j, k}) = normalized;
                }
            }
        }
        return result;
    }
};
