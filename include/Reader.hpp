#pragma once

#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <tuple>
#include <cstdint>
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <unordered_map>

#include "Tensor.hpp"
#include "Tokenizer.hpp"

using namespace std;

class Reader {
public:
    static std::tuple<std::vector<Tensor>, std::vector<Tensor>> load_bin(
        const std::string& filename,
        int num_samples,
        const std::vector<size_t>& x_shape = {1, 1, 28, 28},
        const std::vector<size_t>& y_shape = {10} ) {
        ifstream file(filename, ios::binary);
        if (!file.is_open()) {
            throw runtime_error("No se pudo abrir el archivo binario: " + filename);
        }

        // Leer cabecera
        int32_t header[3];
        file.read(reinterpret_cast<char*>(header), sizeof(header));

        int total_images = header[0];
        int rows = header[1];
        int cols = header[2];
        int image_size = rows * cols;

        if (num_samples > total_images) {
            throw runtime_error("num_samples excede el número de imágenes disponibles.");
        }

        std::vector<Tensor> X, Y;
        X.reserve(num_samples);
        Y.reserve(num_samples);

        for (int i = 0; i < num_samples; ++i) {
            // Leer etiqueta
            unsigned char label;
            file.read(reinterpret_cast<char*>(&label), 1);

            // Leer imagen
            std::vector<float> x_data(image_size);
            for (int j = 0; j < image_size; ++j) {
                unsigned char pixel;
                file.read(reinterpret_cast<char*>(&pixel), 1);
                x_data[j] = static_cast<float>(pixel) / 255.0f;
            }

            // Crear tensor de entrada
            Tensor x_tensor(x_shape);
            x_tensor.data = std::move(x_data);

            // Crear tensor de etiqueta one-hot
            Tensor y_tensor(y_shape);
            y_tensor.data[label] = 1.0f;

            X.push_back(std::move(x_tensor));
            Y.push_back(std::move(y_tensor));
        }

        file.close();
        return {X, Y};
    }
};


class ReaderDataset {
public:
        // Método estático para cargar los datos desde un archivo CSV
    static tuple<vector<Tensor>, vector<Tensor>> load_csv(const string &file_path, const unordered_map<string, int>& vocab) {
        vector<Tensor> X;  // Almacenará las reseñas tokenizadas
        vector<Tensor> Y;  // Almacenará las etiquetas (0 = negativa, 1 = positiva)

        ifstream file(file_path);
        if (!file.is_open()) {
            throw runtime_error("No se pudo abrir el archivo: " + file_path);
        }

        string line;
        while (getline(file, line)) {
            stringstream ss(line);
            string text, label_str;
            
            // Leer la línea y separar la reseña y la etiqueta
            getline(ss, text, ',');  // Reseña
            getline(ss, label_str, '\n');  // Etiqueta
            
            // Convertir la etiqueta a 0 (negativa) o 1 (positiva)
            int label = (label_str == "positive") ? 1 : 0;

            // Tokenizar el texto y convertirlo a tensor
            Tokenizer tokenizer(vocab);
            vector<int> token_ids = tokenizer.tokenize(text);
            Tensor tensor = tokenizer.tokensToTensor(token_ids);

            // Almacenar el tensor de la reseña
            X.push_back(tensor);

            // Almacenar el tensor de la etiqueta
            Tensor label_tensor({1});
            label_tensor({0}) = label;
            Y.push_back(label_tensor);
        }

        file.close();
        return make_tuple(X, Y);
    }
};


// Función para preprocesar el texto (convertir a minúsculas y eliminar puntuación)
string preprocessText(const string &text) {
    string processed_text;
    for (char c : text) {
        if (isalpha(c) || c == ' ') {
            processed_text.push_back(tolower(c)); // Convierte a minúsculas
        }
    }
    return processed_text;
}

// Función para cortar palabras en sub-palabras (substrings) basados en n-gramas
vector<string> generateSubwords(const string &text) {
    vector<string> subwords;
    stringstream ss(text);
    string word;
    
    while (ss >> word) {
        // Cortar la palabra en sub-palabras (n-gramas)
        for (size_t i = 1; i <= word.size(); ++i) {
            for (size_t j = 0; j <= word.size() - i; ++j) {
                subwords.push_back(word.substr(j, i));
            }
        }
    }
    return subwords;
}

// Función para construir el vocabulario
unordered_map<string, int> buildVocabulary(const vector<string> &subwords) {
    unordered_map<string, int> vocab;
    for (const string &subword : subwords) {
        vocab[subword]++;
    }
    return vocab;
}

// Función para imprimir el vocabulario
void printVocabulary(const unordered_map<string, int> &vocab) {
    for (const auto &entry : vocab) {
        cout << entry.first << ": " << entry.second << endl;
    }
}

// Función para guardar el vocabulario en un archivo binario
void saveVocabulary(const unordered_map<string, int>& vocab, const string& file_path) {
    ofstream file(file_path, ios::binary);
    if (!file.is_open()) {
        cerr << "Error al abrir el archivo para guardar el vocabulario." << endl;
        return;
    }

    // Guardar el tamaño del vocabulario
    size_t size = vocab.size();
    file.write(reinterpret_cast<const char*>(&size), sizeof(size));

    // Guardar las palabras y sus índices
    for (const auto& entry : vocab) {
        size_t word_size = entry.first.size();
        file.write(reinterpret_cast<const char*>(&word_size), sizeof(word_size));
        file.write(entry.first.c_str(), word_size);
        file.write(reinterpret_cast<const char*>(&entry.second), sizeof(entry.second));
    }

    file.close();
    cout << "Vocabulario guardado en: " << file_path << endl;
}

// Función para cargar el vocabulario desde un archivo binario
unordered_map<string, int> loadVocabulary(const string& file_path) {
    unordered_map<string, int> vocab;
    ifstream file(file_path, ios::binary);
    if (!file.is_open()) {
        cerr << "Error al abrir el archivo para cargar el vocabulario." << endl;
        return vocab;
    }

    // Leer el tamaño del vocabulario
    size_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size));

    // Leer las palabras y sus índices
    for (size_t i = 0; i < size; ++i) {
        size_t word_size;
        file.read(reinterpret_cast<char*>(&word_size), sizeof(word_size));

        string word(word_size, '\0');
        file.read(&word[0], word_size);

        int index;
        file.read(reinterpret_cast<char*>(&index), sizeof(index));

        vocab[word] = index;
    }

    file.close();
    cout << "Vocabulario cargado desde: " << file_path << endl;
    return vocab;
}