#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include "Tensor.hpp"

using namespace std;

class MultiHeadAttention {
public:
    // Constructor: Inicializa las matrices de pesos para las 3 proyecciones (Q, K, V) y la proyección final
    MultiHeadAttention(size_t num_heads, size_t embedding_dim) 
        : num_heads(num_heads), embedding_dim(embedding_dim) {

        if (embedding_dim % num_heads != 0) {
            throw runtime_error("El tamaño del embedding debe ser divisible por el número de cabezas.");
        }

        head_dim = embedding_dim / num_heads;  // Dimensión de cada cabeza de atención

        // Inicializar las matrices de pesos para Q, K, V (dim: [embedding_dim, head_dim])
        Wq = Tensor({embedding_dim, head_dim});
        Wk = Tensor({embedding_dim, head_dim});
        Wv = Tensor({embedding_dim, head_dim});
        Wout = Tensor({embedding_dim, embedding_dim});  // Proyección final para concatenar las cabezas

        // Inicializamos los pesos con valores pequeños (por ejemplo, ceros)
        Wq.fill(0.01);
        Wk.fill(0.01);
        Wv.fill(0.01);
        Wout.fill(0.01);
    }

    // Forward Pass: Recibe los embeddings y devuelve la salida de la atención multi-cabeza
    Tensor forward(const Tensor &input) {
        size_t batch_size = input.shape[0];  // Número de ejemplos en el lote
        size_t sequence_length = input.shape[1];  // Longitud de la secuencia (número de tokens)

        // Proyección de las entradas a Q, K, V
        Tensor Q = input * Wq;  // [batch_size, sequence_length, embedding_dim]
        Tensor K = input * Wk;  // [batch_size, sequence_length, embedding_dim]
        Tensor V = input * Wv;  // [batch_size, sequence_length, embedding_dim]

        // Dividir Q, K, V en múltiples cabezas
        Q = split_heads(Q);  // [batch_size, num_heads, sequence_length, head_dim]
        K = split_heads(K);  // [batch_size, num_heads, sequence_length, head_dim]
        V = split_heads(V);  // [batch_size, num_heads, sequence_length, head_dim]

        // Cálculo de la atención para cada cabeza
        Tensor attention_output = scaled_dot_product_attention(Q, K, V);  // [batch_size, num_heads, sequence_length, head_dim]

        // Concatenar las cabezas de atención
        attention_output = concat_heads(attention_output);  // [batch_size, sequence_length, embedding_dim]

        // Aplicar la proyección final
        attention_output = attention_output * Wout;  // [batch_size, sequence_length, embedding_dim]

        return attention_output;
    }

private:
    size_t num_heads;        // Número de cabezas de atención
    size_t embedding_dim;    // Dimensión de los embeddings (768 para DistilBERT)
    size_t head_dim;         // Dimensión de cada cabeza (embedding_dim / num_heads)

    Tensor Wq, Wk, Wv, Wout; // Matrices de pesos para Q, K, V y la proyección final

    // Divide los tensores Q, K, V en múltiples cabezas
    Tensor split_heads(const Tensor &tensor) {
        size_t batch_size = tensor.shape[0];
        size_t sequence_length = tensor.shape[1];
        
        // Reshape: [batch_size, sequence_length, num_heads, head_dim]
        Tensor reshaped({batch_size, sequence_length, num_heads, head_dim});
        
        // Aquí asumimos que los tensores Q, K, V ya tienen la forma [batch_size, sequence_length, embedding_dim]
        // Debemos reorganizarlos en la forma [batch_size, num_heads, sequence_length, head_dim]
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < sequence_length; ++j) {
                for (size_t h = 0; h < num_heads; ++h) {
                    for (size_t d = 0; d < head_dim; ++d) {
                        reshaped({i, j, h, d}) = tensor({i, j, h * head_dim + d});
                    }
                }
            }
        }

        return reshaped;
    }

    // Realiza la atención escalada de producto punto entre Q, K y V
    Tensor scaled_dot_product_attention(const Tensor &Q, const Tensor &K, const Tensor &V) {
        size_t batch_size = Q.shape[0];
        size_t sequence_length = Q.shape[2];
        
        // Calculamos la atención de producto punto escalado
        Tensor scores = matmul(Q, K.transpose());  // [batch_size, num_heads, sequence_length, sequence_length]
        scores = scores * (1.0f / sqrt(head_dim)); // Escalado por la raíz de head_dim
        
        // Aplicar softmax a las puntuaciones para obtener los pesos de atención
        Tensor attention_weights = softmax(scores); // [batch_size, num_heads, sequence_length, sequence_length]

        // Multiplicar los pesos de atención por los valores V
        Tensor output = matmul(attention_weights, V); // [batch_size, num_heads, sequence_length, head_dim]
        
        return output;
    }

    // Concatenamos las cabezas de atención para obtener el resultado final
    Tensor concat_heads(const Tensor &tensor) {
        size_t batch_size = tensor.shape[0];
        size_t sequence_length = tensor.shape[1];

        // Reshape a [batch_size, sequence_length, embedding_dim]
        Tensor reshaped({batch_size, sequence_length, embedding_dim});
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < sequence_length; ++j) {
                for (size_t h = 0; h < num_heads; ++h) {
                    for (size_t d = 0; d < head_dim; ++d) {
                        reshaped({i, j, h * head_dim + d}) = tensor({i, j, h, d});
                    }
                }
            }
        }

        return reshaped;
    }

    // Función softmax para normalizar las puntuaciones de atención
    Tensor softmax(const Tensor &tensor) {
        // Asumimos que softmax se aplica en la última dimensión (sequence_length)
        Tensor result(tensor.shape);
        for (size_t i = 0; i < tensor.shape[0]; ++i) {
            for (size_t j = 0; j < tensor.shape[1]; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < tensor.shape[2]; ++k) {
                    sum += exp(tensor({i, j, k}));
                }
                for (size_t k = 0; k < tensor.shape[2]; ++k) {
                    result({i, j, k}) = exp(tensor({i, j, k})) / sum;
                }
            }
        }
        return result;
    }

    // Multiplicación de matrices (producto punto)
    Tensor matmul(const Tensor &A, const Tensor &B) {
        // Realiza el producto punto entre dos tensores. Implementación simplificada.
        if (A.shape[2] != B.shape[1]) {
            throw runtime_error("Las dimensiones no son compatibles para multiplicación de matrices.");
        }

        Tensor result({A.shape[0], A.shape[1], B.shape[2]});
        for (size_t i = 0; i < A.shape[0]; ++i) {
            for (size_t j = 0; j < A.shape[1]; ++j) {
                for (size_t k = 0; k < B.shape[2]; ++k) {
                    result({i, j, k}) = 0.0f;
                    for (size_t l = 0; l < A.shape[2]; ++l) {
                        result({i, j, k}) += A({i, j, l}) * B({i, l, k});
                    }
                }
            }
        }

        return result;
    }
};
