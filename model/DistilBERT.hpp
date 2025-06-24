#pragma once

#include "Tensor.hpp"
#include "MultiHeadAttention.hpp"
#include "FeedForward.hpp"
#include "LayerNormalization.hpp"
#include "TransformerEncoderLayer.hpp"
#include <vector>

class DistilBERT {
public:
    // Constructor: Inicializa los componentes de DistilBERT
    DistilBERT(size_t embedding_dim, size_t num_heads, size_t hidden_dim, size_t num_layers)
        : embedding_dim(embedding_dim), num_heads(num_heads), hidden_dim(hidden_dim), num_layers(num_layers) {
        
        // Crear las capas de Transformer Encoder Layer
        for (size_t i = 0; i < num_layers; ++i) {
            encoder_layers.push_back(TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim));
        }
        
        // Capa final de clasificación (usaremos el token [CLS] para la clasificación)
        W_out = Tensor({embedding_dim, 2});  // Proyección a 2 clases (positiva/negativa)
        b_out = Tensor({2});  // Vector de sesgo para la clasificación
        
        // Inicializar los pesos de salida con pequeños valores
        W_out.fill(0.01);
        b_out.fill(0.01);
    }

    // Forward Pass: Recibe los embeddings y realiza el pase hacia adelante
    int forward(const Tensor &input) {
        // Paso 1: Embeddings (token embeddings + positional embeddings)
        Tensor embeddings = input;  // Asumimos que los embeddings ya han sido creados antes de entrar aquí

        // Paso 2: Pasar por las capas del Transformer Encoder
        Tensor output = embeddings;
        for (size_t i = 0; i < num_layers; ++i) {
            output = encoder_layers[i].forward(output);  // Aplicamos cada capa del encoder
        }

        // Paso 3: Usar la salida del token [CLS] para la clasificación
        Tensor cls_token_output = output.slice({0, 0});  // Tomamos el primer token de la secuencia [CLS]
        
        // Paso 4: Proyección final para clasificación
        Tensor logits = cls_token_output * W_out + b_out;  // [batch_size, 2]
        
        // Paso 5: Aplicar softmax para obtener probabilidades
        Tensor probabilities = softmax(logits);

        // Paso 6: Obtener la predicción (clasificación binaria: 0 = negativo, 1 = positivo)
        return probabilities.data[0] > probabilities.data[1] ? 0 : 1;
    }

private:
    size_t embedding_dim, num_heads, hidden_dim, num_layers;
    std::vector<TransformerEncoderLayer> encoder_layers;
    Tensor W_out, b_out;

    // Softmax para normalizar las puntuaciones de clasificación
    Tensor softmax(const Tensor &tensor) {
        Tensor result(tensor.shape);
        float sum = 0.0f;
        
        // Calcular la suma de las exponenciales
        for (size_t i = 0; i < tensor.shape[1]; ++i) {
            sum += exp(tensor({0, i}));
        }
        
        // Aplicar la función softmax
        for (size_t i = 0; i < tensor.shape[1]; ++i) {
            result({0, i}) = exp(tensor({0, i})) / sum;
        }
        
        return result;
    }
};