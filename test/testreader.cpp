#include "Reader.hpp"
#include "Tokenizer.hpp"
#include "Tensor.hpp"

#include <iostream>
#include <unordered_map>

using namespace std;


// Ejemplo de uso
int buildAndSaveVocabulary() {
    // Leer un archivo con las reseñas de IMDB
    ifstream file("../datasets/IMDB_Dataset.csv");
    if (!file.is_open()) {
        cerr << "Error al abrir el archivo." << endl;
        return -1;
    }

    unordered_map<string, int> global_vocab;
    string line;

    int i = 1;
    while (getline(file, line)) {
        // Preprocesar el texto (eliminando puntuaciones y convirtiendo a minúsculas)
        string processed_text = preprocessText(line);

        // Generar sub-palabras para el texto preprocesado
        vector<string> subwords = generateSubwords(processed_text);

        // Construir el vocabulario
        unordered_map<string, int> vocab = buildVocabulary(subwords);

        // Combinar el vocabulario global con el vocabulario local
        for (const auto &entry : vocab) {
            global_vocab[entry.first] += entry.second;
        }
        if (i%1000==0 || i==0){
            cout << "line " << i << "\n";
        }
        i++;
    }

    file.close();

    saveVocabulary(global_vocab, "vocabulary.bin");

    return 0;
}


int test2() {
    // Crear un vocabulario de ejemplo (en un caso real, debería cargarse desde un archivo)
    unordered_map<string, int> vocab = {
        {"[CLS]", 0},
        {"[SEP]", 1},
        {"[UNK]", 2},
        {"hello", 3},
        {"world", 4},
        {"##s", 5},
        {"##d", 6}
    };

    // Cargar el dataset IMDB
    string file_path = "../datasets/IMDB_Dataset.csv";
    auto [X, Y] = ReaderDataset::load_csv(file_path, vocab);

    cout << "X size: " << X.size() << endl;
    cout << "Y size: " << Y.size() << endl;

    // Imprimir algunos resultados
    cout << "Número de reseñas cargadas: " << X.size() << endl;
    cout << "Número de labels cargados: " << Y.size() << endl;
    cout << "Tensor de la primera reseña: " << endl;
    cout << X[0] << endl;
    cout << "Etiqueta de la primera reseña: " << Y[0] << endl;

    return 0;
}


int main(){
    // buildAndSaveVocabulary();
    // return 0;
    
    // Cargar el vocabulario desde el archivo binario
    unordered_map<string, int> loaded_vocab = loadVocabulary("vocabulary.bin");

    // Crear un tokenizador con el vocabulario cargado
    Tokenizer tokenizer(loaded_vocab);

    // Texto de ejemplo
    string text = "If anyone is wondering why no one makes movies like they used to, with conversation, character";

    // Tokenizar el texto
    vector<int> token_ids = tokenizer.tokenize(text);

    // Mostrar los resultados de tokenización
    cout << "Tokens (ID's): ";
    cout << token_ids << endl;

    // Convertir los tokens a tensor
    Tensor tensor = tokenizer.tokensToTensor(token_ids);

    // Imprimir el tensor
    cout << "Tensor de Tokens: " << endl;
    cout << tensor << endl;

    return 0;
}