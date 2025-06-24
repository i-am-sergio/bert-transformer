#include <iostream>
#include <vector>
#include "DistilBERT.hpp"
#include "Tensor.hpp"

using namespace std;

// Simula la carga del dataset IMDB (esto es solo un ejemplo).
void load_imdb_data(vector<Tensor>& train_data, vector<int>& train_labels, vector<Tensor>& test_data, vector<int>& test_labels) {
    // Simulación de datos (en la práctica, aquí deberías cargar y tokenizar las reseñas IMDB)
    // Asumimos que el tamaño de embedding es 768 y la secuencia de tokens tiene longitud 128
    train_data.resize(100);  // 100 ejemplos en el entrenamiento
    train_labels.resize(100);
    test_data.resize(20);    // 20 ejemplos en el test
    test_labels.resize(20);

    // Rellenamos los datos con valores ficticios
    for (size_t i = 0; i < 100; ++i) {
        train_data[i] = Tensor({1, 128, 768});  // Un ejemplo con tamaño 128x768
        train_labels[i] = (i % 2);  // Alterna entre 0 y 1 para clasificación binaria
    }
    for (size_t i = 0; i < 20; ++i) {
        test_data[i] = Tensor({1, 128, 768});  // Un ejemplo con tamaño 128x768
        test_labels[i] = (i % 2);  // Alterna entre 0 y 1 para clasificación binaria
    }
}

// Entrenamiento del modelo por un número de épocas
void train(DistilBERT &model, vector<Tensor>& train_data, vector<int>& train_labels, int num_epochs) {
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float total_loss = 0.0;
        int correct_predictions = 0;

        // Simula el entrenamiento (en la práctica, deberías optimizar los pesos con un optimizador)
        for (size_t i = 0; i < train_data.size(); ++i) {
            // Forward pass (predicción)
            int prediction = model.forward(train_data[i]);

            // Cálculo de la pérdida (usamos una simple comparación para la pérdida binaria)
            int true_label = train_labels[i];
            total_loss += (prediction == true_label) ? 0.0f : 1.0f;

            // Cálculo de las predicciones correctas
            if (prediction == true_label) {
                correct_predictions++;
            }

            // Aquí podrías actualizar los pesos del modelo (optimización) después de cada paso,
            // pero como ejemplo, no estamos haciendo un paso de optimización real aquí.
        }

        // Reporte de la pérdida y precisión por época
        float accuracy = (float)correct_predictions / train_data.size();
        cout << "Epoch " << epoch + 1 << " | Loss: " << total_loss / train_data.size() << " | Accuracy: " << accuracy * 100 << "%" << endl;
    }
}

// Evaluación del modelo en el conjunto de prueba
void evaluate(DistilBERT &model, vector<Tensor>& test_data, vector<int>& test_labels) {
    int correct_predictions = 0;

    // Simula la evaluación (en la práctica, deberías aplicar el modelo a todo el conjunto de prueba)
    for (size_t i = 0; i < test_data.size(); ++i) {
        int prediction = model.forward(test_data[i]);
        if (prediction == test_labels[i]) {
            correct_predictions++;
        }
    }

    float accuracy = (float)correct_predictions / test_data.size();
    cout << "Test Accuracy: " << accuracy * 100 << "%" << endl;
}

int main() {
    // Definir el tamaño de los embeddings y los parámetros de DistilBERT
    size_t embedding_dim = 768;  // Dimensión de los embeddings (similar a DistilBERT)
    size_t num_heads = 8;        // Número de cabezas en la multi-head attention
    size_t hidden_dim = 2048;    // Tamaño de la capa oculta en el FFN
    size_t num_layers = 6;       // Número de capas de Transformer Encoder (DistilBERT tiene 6 capas)

    // Crear el modelo DistilBERT
    DistilBERT model(embedding_dim, num_heads, hidden_dim, num_layers);

    // Cargar el dataset IMDB (simulado)
    vector<Tensor> train_data, test_data;
    vector<int> train_labels, test_labels;
    load_imdb_data(train_data, train_labels, test_data, test_labels);

    // Entrenar el modelo por 5 épocas
    int num_epochs = 5;
    train(model, train_data, train_labels, num_epochs);

    // Evaluar el modelo en el conjunto de prueba
    evaluate(model, test_data, test_labels);

    return 0;
}
