#pragma once

#include "Tensor.hpp"
#include <vector>
#include <random>
#include <string>
#include <iostream>

using namespace std;

class EmbeddingLayer {
private:
    Tensor embeddings;   // Tensor de embeddings, de tamaño [vocab_size, embedding_dim]
    size_t vocab_size;   // Tamaño del vocabulario
    size_t embedding_dim; // Dimensión de los embeddings

public:
    // Constructor: Inicializa los embeddings aleatoriamente
    EmbeddingLayer(size_t vocab_size_, size_t embedding_dim_) 
        : vocab_size(vocab_size_), embedding_dim(embedding_dim_) {

        // Inicializar los embeddings con una distribución normal
        embeddings = Tensor({vocab_size, embedding_dim});
        initialize_embeddings();
    }

    // Inicialización de los embeddings con una distribución normal
    void initialize_embeddings() {
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<float> dist(0.0f, 0.02f); // Media 0, desviación estándar 0.02

        for (size_t i = 0; i < vocab_size; ++i) {
            for (size_t j = 0; j < embedding_dim; ++j) {
                embeddings({i, j}) = dist(gen);  // Inicializa los valores de los embeddings
            }
        }
    }

    // Forward pass: Convierte los índices de tokens a sus embeddings correspondientes
    Tensor forward(const Tensor &input) {
        size_t batch_size = input.shape[0];   // Tamaño del lote
        size_t seq_len = input.shape[1];      // Longitud de la secuencia

        // Creamos un tensor de salida con la forma [batch_size, seq_len, embedding_dim]
        Tensor output({batch_size, seq_len, embedding_dim});
        
        // Mapear los índices de tokens a embeddings
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < seq_len; ++j) {
                size_t token_id = static_cast<size_t>(input({i, j}));  // Obtener el índice del token
                // Copiar el embedding correspondiente al token
                for (size_t k = 0; k < embedding_dim; ++k) {
                    output({i, j, k}) = embeddings({token_id, k});
                }
            }
        }

        return output;
    }

    // Backward pass: Calcula los gradientes de los embeddings
    Tensor backward(const Tensor &grad_output, const Tensor &input) {
        // Los gradientes de los embeddings se suman por token
        Tensor grad_embeddings({vocab_size, embedding_dim});
        grad_embeddings.fill(0.0f);  // Inicializamos los gradientes a cero

        size_t batch_size = grad_output.shape[0];
        size_t seq_len = grad_output.shape[1];

        // Calcular gradientes de embeddings sumando gradientes de cada token
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < seq_len; ++j) {
                size_t token_id = static_cast<size_t>(input({i, j}));  // Obtener el índice del token de la entrada
                for (size_t k = 0; k < embedding_dim; ++k) {
                    grad_embeddings({token_id, k}) += grad_output({i, j, k});  // Sumar gradientes
                }
            }
        }

        return grad_embeddings;  // Retornar los gradientes de los embeddings
    }
    
    // Actualizar los embeddings usando el optimizador
    void update_embeddings(const Tensor &grad_embeddings, float learning_rate) {
        for (size_t i = 0; i < vocab_size; ++i) {
            for (size_t j = 0; j < embedding_dim; ++j) {
                embeddings({i, j}) -= learning_rate * grad_embeddings({i, j});  // Actualización simple
            }
        }
    }

    // Función para imprimir los embeddings
    void print_embeddings() {
        cout << "Embeddings: " << endl;
        for (size_t i = 0; i < vocab_size; ++i) {
            for (size_t j = 0; j < embedding_dim; ++j) {
                cout << embeddings({i, j}) << " ";
            }
            cout << endl;
        }
    }
};
