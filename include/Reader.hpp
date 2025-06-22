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