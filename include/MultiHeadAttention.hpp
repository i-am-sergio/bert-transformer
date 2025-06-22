#pragma once

#include "Tensor.hpp"
#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <algorithm>

using namespace std;

class MultiHeadAttention {
private:
    size_t num_heads;          // Número de cabezas de atención
    size_t embedding_dim;      // Dimensión de los embeddings
    size_t head_dim;           // Dimensión de cada cabeza de atención (embedding_dim / num_heads)
    Tensor Wq, Wk, Wv, Wo;     // Pesos de las proyecciones Q, K, V y la salida
    float scale;               // Factor de escala (1/sqrt(d_k))

public:
    // Constructor: Inicializa los pesos de la atención múltiple
    MultiHeadAttention(size_t num_heads_, size_t embedding_dim_) 
        : num_heads(num_heads_), embedding_dim(embedding_dim_) {
        
        head_dim = embedding_dim / num_heads;
        scale = 1.0f / sqrt(head_dim);

        // Inicialización de los pesos de proyección para Q, K, V y la salida
        Wq = Tensor({embedding_dim, embedding_dim});
        Wk = Tensor({embedding_dim, embedding_dim});
        Wv = Tensor({embedding_dim, embedding_dim});
        Wo = Tensor({embedding_dim, embedding_dim});

        initialize_weights();
    }

    // Inicialización de los pesos con distribución normal
    void initialize_weights() {
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<float> dist(0.0f, 0.02f);  // Media 0, desviación estándar 0.02

        // Inicializar Q, K, V y O con distribución normal
        initialize_tensor(Wq, dist, gen);
        initialize_tensor(Wk, dist, gen);
        initialize_tensor(Wv, dist, gen);
        initialize_tensor(Wo, dist, gen);
    }

    // Inicializar los valores del tensor con la distribución dada
    void initialize_tensor(Tensor &tensor, normal_distribution<float> &dist, mt19937 &gen) {
        for (size_t i = 0; i < tensor.shape[0]; ++i) {
            for (size_t j = 0; j < tensor.shape[1]; ++j) {
                tensor({i, j}) = dist(gen);
            }
        }
    }

    // Scaled dot-product attention
    Tensor scaled_dot_product_attention(const Tensor &Q, const Tensor &K, const Tensor &V) {
        size_t batch_size = Q.shape[0];
        size_t seq_len = Q.shape[1];
        size_t head_dim = Q.shape[2];
        
        // QK^T
        Tensor scores({batch_size, seq_len, seq_len});
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = 0; j < seq_len; ++j) {
                    float sum = 0;
                    for (size_t k = 0; k < head_dim; ++k) {
                        sum += Q({b, i, k}) * K({b, j, k});
                    }
                    scores({b, i, j}) = sum * scale;
                }
            }
        }
        
        // Softmax
        Tensor weights = softmax(scores);
        
        // Multiplicar por V
        Tensor output({batch_size, seq_len, head_dim});
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = 0; j < head_dim; ++j) {
                    float sum = 0;
                    for (size_t k = 0; k < seq_len; ++k) {
                        sum += weights({b, i, k}) * V({b, k, j});
                    }
                    output({b, i, j}) = sum;
                }
            }
        }
        
        return output;
    }

    // Forward pass: Realiza la atención múltiple
    Tensor forward(const Tensor &input) {
        size_t batch_size = input.shape[0];
        size_t seq_len = input.shape[1];

        // Proyectar Q, K, V
        Tensor Q = matmul(input, Wq);  // [batch, seq_len, emb_dim]
        Tensor K = matmul(input, Wk);
        Tensor V = matmul(input, Wv);

        // Dividir en cabezas
        vector<Tensor> Q_heads = split(Q);
        vector<Tensor> K_heads = split(K);
        vector<Tensor> V_heads = split(V);

        // Procesar cada cabeza
        vector<Tensor> head_outputs(num_heads);
        for (size_t h = 0; h < num_heads; ++h) {
            head_outputs[h] = scaled_dot_product_attention(Q_heads[h], K_heads[h], V_heads[h]);
        }

        // Concatenar
        Tensor output = concat_heads(head_outputs);

        // Proyección final
        output = matmul(output, Wo);

        return output;
    }

    // Matmul para multiplicar tensores
    Tensor matmul(const Tensor &a, const Tensor &b) {
        // Asumimos que a es [batch, m, n] y b es [n, p]
        size_t batch = a.shape[0];
        size_t m = a.shape[1];
        size_t n = a.shape[2];
        size_t p = b.shape[1];
        
        Tensor result({batch, m, p});
        
        for (size_t b_idx = 0; b_idx < batch; ++b_idx) {
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < p; ++j) {
                    float sum = 0;
                    for (size_t k = 0; k < n; ++k) {
                        sum += a({b_idx, i, k}) * b({k, j});
                    }
                    result({b_idx, i, j}) = sum;
                }
            }
        }
        
        return result;
    }

    // Softmax para calcular la atención
    Tensor softmax(const Tensor &input) {
        Tensor result(input.shape);
        float max_value = *max_element(input.data.begin(), input.data.end());
        float sum_exp = 0.0f;

        // Exponencial y normalización
        for (size_t i = 0; i < input.get_size(); ++i) {
            result.data[i] = exp(input.data[i] - max_value);
            sum_exp += result.data[i];
        }

        for (float &val : result.data) {
            val /= sum_exp;
        }

        return result;
    }

    // Dividir el tensor en múltiples cabezas
    vector<Tensor> split(const Tensor &input) {
        size_t batch_size = input.shape[0];
        size_t seq_len = input.shape[1];
        size_t head_dim = embedding_dim / num_heads;

        vector<Tensor> heads(num_heads);

        for (size_t h = 0; h < num_heads; ++h) {
            heads[h] = Tensor({batch_size, seq_len, head_dim});
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t s = 0; s < seq_len; ++s) {
                    for (size_t d = 0; d < head_dim; ++d) {
                        size_t emb_idx = h * head_dim + d;
                        heads[h]({b, s, d}) = input({b, s, emb_idx});
                    }
                }
            }
        }

        return heads;
    }

    // Concatenar los resultados de todas las cabezas
    Tensor concat_heads(const vector<Tensor> &heads) {
        size_t batch_size = heads[0].shape[0];
        size_t seq_len = heads[0].shape[1];
        size_t head_dim = heads[0].shape[2];
        size_t emb_dim = num_heads * head_dim;

        Tensor output({batch_size, seq_len, emb_dim});

        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t s = 0; s < seq_len; ++s) {
                for (size_t h = 0; h < num_heads; ++h) {
                    for (size_t d = 0; d < head_dim; ++d) {
                        size_t emb_idx = h * head_dim + d;
                        output({b, s, emb_idx}) = heads[h]({b, s, d});
                    }
                }
            }
        }

        return output;
    }
};