#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <sstream>
#include <iostream>
#include <algorithm>
#include "Tensor.hpp"

using namespace std;

class Tokenizer {
public:
    Tokenizer(const unordered_map<string, int>& vocab_) : vocab(vocab_) {
        // Verificamos si el vocabulario tiene un token UNK, si no, lo añadimos con el valor correspondiente
        if (vocab.find("[UNK]") == vocab.end()) {
            vocab["[UNK]"] = 2;  // Definimos 2 como el índice de [UNK] por defecto
        }
    }

    // Tokeniza el texto y convierte los tokens a índices de vocabulario
    vector<int> tokenize(const string &text) {
        vector<string> tokens = wordPieceTokenize(text);
        vector<int> token_ids;
        
        for (const auto &token : tokens) {
            auto it = vocab.find(token);
            if (it != vocab.end()) {
                token_ids.push_back(it->second);  // Si el token está en el vocabulario, agregamos su índice
            } else {
                // Si el token no está en el vocabulario, usamos el índice de [UNK]
                token_ids.push_back(vocab.at("[UNK]"));
            }
        }

        return token_ids;
    }

    // Convierte una lista de tokens en un tensor
    Tensor tokensToTensor(const vector<int> &tokens) {
        vector<size_t> shape = {tokens.size()};
        Tensor tensor(shape);
        for (size_t i = 0; i < tokens.size(); ++i) {
            tensor({i}) = tokens[i];
        }
        return tensor;
    }

private:
    unordered_map<string, int> vocab;  // Mapa del vocabulario, que asigna cada palabra a un índice

    // Tokeniza el texto usando el algoritmo WordPiece
    vector<string> wordPieceTokenize(const string &text) {
        vector<string> tokens;
        stringstream ss(text);
        string word;
        
        while (ss >> word) {
            tokens.push_back(applyWordPiece(word));
        }
        
        return tokens;
    }

    // Aplica el tokenizador WordPiece a una palabra
    string applyWordPiece(const string &word) {
        // Por simplicidad, consideramos que cada palabra se divide en subpalabras de manera simple.
        // En un modelo real de WordPiece, se usarían algoritmos de segmentación más complejos.
        
        string wordPieceToken = word;
        if (vocab.find(word) != vocab.end()) {
            return wordPieceToken;
        } else {
            // Aquí aplicamos un enfoque simple donde dividimos las palabras en sub-palabras.
            wordPieceToken = word.substr(0, word.length() - 1) + "##" + word.back();
            return wordPieceToken;
        }
    }
};
