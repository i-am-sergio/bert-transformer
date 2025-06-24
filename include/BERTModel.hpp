#pragma once

#include "MultiHeadAttention.hpp"
#include "TransformerEncoderLayer.hpp"
#include "Tokenizer.hpp"
#include "Tensor.hpp"
#include "Layer.hpp"
#include "Dense.hpp"
#include "LayerNormalization.hpp"
#include "Dropout.hpp"
#include "EmbeddingLayer.hpp"
#include "Optimizer.hpp"
#include "AdamOptimizer.hpp"

#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <random>

class BERTModel {
private:
    // Configuración del modelo
    size_t vocab_size;
    size_t max_seq_length;
    size_t embedding_dim;
    size_t num_heads;
    size_t num_layers;
    size_t ffn_dim;
    float dropout_rate;

    // Componentes del modelo
    Tokenizer tokenizer;
    EmbeddingLayer token_embeddings;
    EmbeddingLayer position_embeddings;
    vector<TransformerEncoderLayer> encoder_layers;
    Dense classifier;
    Dropout dropout;
    LayerNormalization layer_norm;

    // Métodos auxiliares
    Tensor create_position_ids(size_t seq_length) {
        Tensor positions({1, seq_length});
        for (size_t i = 0; i < seq_length; ++i) {
            positions({0, i}) = static_cast<float>(i);
        }
        return positions;
    }

public:
    BERTModel(const string& vocab_file, 
              size_t vocab_size_ = 30522,
              size_t max_seq_length_ = 512,
              size_t embedding_dim_ = 768,
              size_t num_heads_ = 12,
              size_t num_layers_ = 12,
              size_t ffn_dim_ = 3072,
              float dropout_rate_ = 0.1f)
        : vocab_size(vocab_size_),
          max_seq_length(max_seq_length_),
          embedding_dim(embedding_dim_),
          num_heads(num_heads_),
          num_layers(num_layers_),
          ffn_dim(ffn_dim_),
          dropout_rate(dropout_rate_),
          tokenizer(vocab_file),
          token_embeddings(vocab_size_, embedding_dim_),
          position_embeddings(max_seq_length_, embedding_dim_),
          classifier(embedding_dim_, 2), // 2 clases: positivo/negativo
          dropout(dropout_rate_),
          layer_norm(embedding_dim_) {
        
        // Inicializar capas encoder
        for (size_t i = 0; i < num_layers_; ++i) {
            encoder_layers.emplace_back(num_heads_, embedding_dim_, ffn_dim_, dropout_rate_);
        }
    }

    // Preprocesamiento de texto
    Tensor preprocess(const string& text) {
        // Tokenizar
        vector<string> tokens = tokenizer.tokenize(text);
        tokens = tokenizer.truncate_sequence(tokens, max_seq_length - 2); // Dejar espacio para [CLS] y [SEP]
        
        // Añadir tokens especiales
        tokens.insert(tokens.begin(), "[CLS]");
        tokens.push_back("[SEP]");
        
        // Convertir a IDs
        Tensor input_ids = tokenizer.convert_tokens_to_ids(tokens);
        
        // Añadir padding si es necesario
        if (tokens.size() < max_seq_length) {
            Tensor padded_ids({1, max_seq_length});
            for (size_t i = 0; i < tokens.size(); ++i) {
                padded_ids({0, i}) = input_ids({i});
            }
            for (size_t i = tokens.size(); i < max_seq_length; ++i) {
                padded_ids({0, i}) = 0; // [PAD] token
            }
            input_ids = padded_ids;
        }
        
        return input_ids;
    }

    // Forward pass del modelo
    Tensor forward(const Tensor& input_ids) {
        size_t batch_size = input_ids.shape[0];
        size_t seq_length = input_ids.shape[1];
        
        // Obtener embeddings de tokens
        Tensor token_embeds = token_embeddings.forward(input_ids);
        
        // Obtener embeddings de posición
        Tensor pos_ids = create_position_ids(seq_length);
        Tensor pos_embeds = position_embeddings.forward(pos_ids);
        
        // Sumar embeddings y aplicar dropout
        Tensor embeddings = token_embeds + pos_embeds;
        embeddings = dropout.forward(embeddings);
        
        // Pasar por las capas encoder
        Tensor hidden_states = embeddings;
        for (auto& layer : encoder_layers) {
            hidden_states = layer.forward(hidden_states);
        }
        
        // Obtener el embedding de [CLS] para clasificación
        Tensor cls_embedding({batch_size, embedding_dim});
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t d = 0; d < embedding_dim; ++d) {
                cls_embedding({b, d}) = hidden_states({b, 0, d}); // [CLS] está en posición 0
            }
        }
        
        // Normalización final
        cls_embedding = layer_norm.forward(cls_embedding);
        
        // Clasificación
        Tensor logits = classifier.forward(cls_embedding);
        
        return logits;
    }

    // Entrenamiento del modelo
    void train(const vector<string>& texts, const vector<int>& labels,
               size_t batch_size = 32, size_t epochs = 3, float learning_rate = 5e-5) {
        // Optimizador
        AdamOptimizer optimizer(learning_rate);

        vector<float> all_params; // falta
        vector<float> all_grads;
    
        
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            cout << "Epoch " << epoch + 1 << "/" << epochs << endl;
            
            // Barajar los datos
            vector<size_t> indices(texts.size());
            iota(indices.begin(), indices.end(), 0);
            shuffle(indices.begin(), indices.end(), default_random_engine());
            
            // Entrenamiento por lotes
            for (size_t batch_start = 0; batch_start < texts.size(); batch_start += batch_size) {
                size_t batch_end = min(batch_start + batch_size, texts.size());
                size_t current_batch_size = batch_end - batch_start;
                
                // Preparar batch
                Tensor batch_input({current_batch_size, max_seq_length});
                Tensor batch_labels({current_batch_size, 2});
                
                for (size_t i = batch_start; i < batch_end; ++i) {
                    size_t batch_idx = i - batch_start;
                    
                    // Preprocesar texto
                    Tensor input_ids = preprocess(texts[indices[i]]);
                    
                    // Añadir al batch
                    for (size_t j = 0; j < max_seq_length; ++j) {
                        batch_input({batch_idx, j}) = input_ids({0, j});
                    }
                    
                    // Crear one-hot labels
                    int label = labels[indices[i]];
                    batch_labels({batch_idx, 0}) = (label == 0) ? 1.0f : 0.0f;
                    batch_labels({batch_idx, 1}) = (label == 1) ? 1.0f : 0.0f;
                }
                
                // Forward pass
                Tensor logits = forward(batch_input);
                
                // Calcular pérdida (cross-entropy)
                float loss = 0.0f;
                Tensor grad_output(logits.shape);
                
                for (size_t i = 0; i < current_batch_size; ++i) {
                    // Softmax manual para estabilidad numérica
                    float max_logit = max(logits({i, 0}), logits({i, 1}));
                    float exp0 = exp(logits({i, 0}) - max_logit);
                    float exp1 = exp(logits({i, 1}) - max_logit);
                    float sum_exp = exp0 + exp1;
                    
                    float prob0 = exp0 / sum_exp;
                    float prob1 = exp1 / sum_exp;
                    
                    // Cross-entropy loss
                    loss += - (batch_labels({i, 0}) * log(prob0 + 1e-10f) + 
                              batch_labels({i, 1}) * log(prob1 + 1e-10f));
                    
                    // Gradiente de la pérdida respecto a los logits
                    grad_output({i, 0}) = prob0 - batch_labels({i, 0});
                    grad_output({i, 1}) = prob1 - batch_labels({i, 1});
                }
                
                loss /= current_batch_size;
                grad_output = grad_output * (1.0f / current_batch_size);
                
                // Backward pass (implementación simplificada)
                // En una implementación real necesitarías implementar backward() para todas las capas
                
                // Actualizar parámetros (simplificado)
                optimizer.update(all_params, all_grads);
                
                if (batch_start % (10 * batch_size) == 0) {
                    cout << "Batch " << batch_start / batch_size << ", Loss: " << loss << endl;
                }
            }
        }
    }

    // Predicción de sentimiento
    int predict(const string& text) {
        Tensor input_ids = preprocess(text);
        input_ids = input_ids.reshape({1, max_seq_length}); // Añadir dimensión de batch
        
        Tensor logits = forward(input_ids);
        
        // Devolver la clase con mayor probabilidad
        return (logits({0, 0}) > logits({0, 1})) ? 0 : 1;
    }
};