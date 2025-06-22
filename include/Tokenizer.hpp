#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include "Tensor.hpp"

using namespace std;

class Tokenizer {
private:
    unordered_map<string, int> vocab; // Mapeo de sub-palabras a índices
    vector<string> rev_vocab;         // Mapeo inverso de índices a sub-palabras
    string unk_token = "[UNK]";       // Token desconocido
    string sep_token = "[SEP]";       // Token de separación
    string cls_token = "[CLS]";       // Token de inicio de secuencia

public:
    // Constructor: Carga el vocabulario desde un archivo
    Tokenizer(const string &vocab_file) {
        load_vocab(vocab_file);
    }

    // Cargar el vocabulario desde un archivo
    void load_vocab(const string &vocab_file) {
        ifstream file(vocab_file);
        if (!file.is_open()) {
            throw runtime_error("No se pudo abrir el archivo de vocabulario.");
        }

        string line;
        int idx = 0;
        while (getline(file, line)) {
            // Asumimos que cada línea tiene una sub-palabra
            vocab[line] = idx;
            rev_vocab.push_back(line);
            idx++;
        }

        file.close();
        cout << "Vocabulario cargado: " << vocab.size() << " sub-palabras." << endl;
    }

    // Tokenizar una frase usando WordPiece
    vector<string> tokenize(const string &text) {
        vector<string> tokens;
        string current_token;
        istringstream stream(text);
        
        while (stream >> current_token) {
            vector<string> word_tokens = wordpiece_tokenize(current_token);
            tokens.insert(tokens.end(), word_tokens.begin(), word_tokens.end());
        }
        return tokens;
    }

    // Dividir una palabra en sub-palabras usando el algoritmo WordPiece
    vector<string> wordpiece_tokenize(const string &word) {
        vector<string> word_tokens;
        string current_subword = "";
        
        if (vocab.find(word) != vocab.end()) {
            word_tokens.push_back(word);  // Si la palabra completa está en el vocabulario
        } else {
            // Intentar dividir la palabra
            int start = 0;
            while (start < word.length()) {
                current_subword = word.substr(start, word.length() - start);
                if (vocab.find(current_subword) != vocab.end()) {
                    word_tokens.push_back(current_subword);
                    break;
                }
                start++;
                if (start == word.length()) {
                    word_tokens.push_back(unk_token); // Si no se puede dividir, usar [UNK]
                }
            }
        }

        return word_tokens;
    }

    // Convertir una secuencia de tokens a índices (usando el vocabulario)
    Tensor convert_tokens_to_ids(const vector<string> &tokens) {
        vector<int> token_ids;
        
        // Agregar [CLS] al inicio y [SEP] al final de la secuencia
        token_ids.push_back(vocab[cls_token]);
        for (const string &token : tokens) {
            if (vocab.find(token) != vocab.end()) {
                token_ids.push_back(vocab[token]);
            } else {
                token_ids.push_back(vocab[unk_token]);  // Si no está en el vocabulario, usar [UNK]
            }
        }
        token_ids.push_back(vocab[sep_token]);

        // Convertir a tensor
        Tensor token_tensor({token_ids.size()});
        for (size_t i = 0; i < token_ids.size(); ++i) {
            token_tensor.data[i] = static_cast<float>(token_ids[i]);
        }

        return token_tensor;
    }

    // Función para imprimir el texto tokenizado
    void print_tokens(const vector<string> &tokens) {
        for (const string &token : tokens) {
            cout << token << " ";
        }
        cout << endl;
    }

    // Función para convertir de índice a token
    string convert_id_to_token(int id) {
        if (id >= 0 && id < rev_vocab.size()) {
            return rev_vocab[id];
        } else {
            return unk_token;  // Si el id está fuera del rango, devolver [UNK]
        }
    }
};
