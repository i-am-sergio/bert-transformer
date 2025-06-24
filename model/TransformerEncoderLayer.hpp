#pragma once

#include <vector>
#include <iostream>
#include <cassert>
#include <cmath>
#include "Tensor.hpp"
#include  "MultiHeadAttention.hpp"
#include "FeedForward.hpp"
#include "LayerNormalization.hpp"

using namespace std;

class TransformerEncoderLayer {
public:
    TransformerEncoderLayer(size_t embedding_dim, size_t num_heads, size_t hidden_dim)
        : attention(num_heads, embedding_dim), 
          feed_forward(embedding_dim, hidden_dim), 
          layer_norm1(embedding_dim), 
          layer_norm2(embedding_dim) {}

    Tensor forward(const Tensor &input) {
        // Atención multi-cabeza
        Tensor attn_output = attention.forward(input);  // [batch_size, sequence_length, embedding_dim]

        // Conexión residual + Layer Normalization (después de la atención)
        attn_output = layer_norm1.forward(input + attn_output);  // [batch_size, sequence_length, embedding_dim]

        // Red neuronal feed-forward
        Tensor ffn_output = feed_forward.forward(attn_output);  // [batch_size, sequence_length, embedding_dim]

        // Conexión residual + Layer Normalization (después de la FFN)
        return layer_norm2.forward(attn_output + ffn_output);  // [batch_size, sequence_length, embedding_dim]
    }

private:
    MultiHeadAttention attention;
    FeedForward feed_forward;
    LayerNormalization layer_norm1, layer_norm2;
};
