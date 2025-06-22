#pragma once

#include "MultiHeadAttention.hpp"
#include "Tensor.hpp"
#include "Layer.hpp"
#include "Dropout.hpp"
#include "DenseT.hpp"
#include "LayerNormalization.hpp"

using namespace std;

class TransformerEncoderLayer {
private:
    MultiHeadAttention attention;      // Capa de atención múltiple
    Dense ffn1, ffn2;                // Red neuronal Feed-Forward
    Dropout dropout_attention;        // Dropout en la atención
    Dropout dropout_ffn;              // Dropout en la FFN
    LayerNormalization norm1, norm2;  // Normalización de capa para cada sub-capa
    float dropout_rate;               // Tasa de dropout

public:
    // Constructor: Inicializa la capa del codificador Transformer
    TransformerEncoderLayer(size_t num_heads, size_t embedding_dim, size_t ffn_dim, float dropout_rate_)
        : attention(num_heads, embedding_dim),
          ffn1(embedding_dim, ffn_dim), ffn2(ffn_dim, embedding_dim),
          dropout_attention(dropout_rate_), dropout_ffn(dropout_rate_), 
          norm1(embedding_dim), norm2(embedding_dim), dropout_rate(dropout_rate_) {}

    // Forward pass: Procesa la entrada a través de la capa de codificación
    Tensor forward(const Tensor &input) {
        const auto& input_shape = input.shape;
        cout << "Input shape: " << input.shape[0] << "x" << input.shape[1] << "x" << input.shape[2] << endl;
        // 1. Self-Attention
        Tensor attn_output = attention.forward(input);
        attn_output = dropout_attention.forward(attn_output);
        
        // Verificación de forma
        if (attn_output.shape != input_shape) {
            throw std::runtime_error("Attention output shape mismatch. Expected " + 
                                   to_string(input_shape[0]) + "x" + to_string(input_shape[1]) + "x" + 
                                   to_string(input_shape[2]) + ", got " + 
                                   to_string(attn_output.shape[0]) + "x" + 
                                   to_string(attn_output.shape[1]) + "x" + 
                                   to_string(attn_output.shape[2]));
        }
        
        // 2. Add & Norm
        Tensor add1 = input + attn_output;
        Tensor norm1_out = norm1.forward(add1);
        
        // 3. FFN (dos capas lineales con activación intermedia)
        Tensor ffn_intermediate = ffn1.forward(norm1_out);
        // Aplicar activación GELU/ReLU aquí si es necesario
        Tensor ffn_output = ffn2.forward(ffn_intermediate);
        ffn_output = dropout_ffn.forward(ffn_output);
        
        // Verificación de forma
        if (ffn_output.shape != input_shape) {
            throw std::runtime_error("FFN output shape mismatch");
        }
        
        // 4. Add & Norm
        Tensor add2 = norm1_out + ffn_output;
        return norm2.forward(add2);
    }
};
