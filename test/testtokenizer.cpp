#include "Tokenizer.hpp"
#include "Tensor.hpp"
#include <iostream>
#include <unordered_map>

using namespace std;

/*
Nota: Este es un ejemplo de prueba con un vocabulario limitado para propósitos demostrativos. En un modelo real de BERT, el vocabulario sería mucho más grande y el tokenizador sería más sofisticado.
*/

int main() {
    // Crear un vocabulario de ejemplo para probar el Tokenizer
    unordered_map<string, int> vocab = {
        {"[CLS]", 0},
        {"[SEP]", 1},
        {"[UNK]", 2},
        {"One", 3},
        {"of", 4},
        {"the", 5},
        {"other", 6},
        {"reviewers", 7},
        {"has", 8},
        {"mentioned", 9},
        {"that", 10},
        {"after", 11},
        {"watching", 12},
        {"just", 13},
        {"1", 14},
        {"Oz", 15},
        {"episode", 16},
        {"you'll", 17},
        {"be", 18},
        {"hooked", 19},
        {"The", 20},
        {"show", 21},
        {"starts", 22},
        {"off", 23},
        {"slow", 24},
        {"but", 25},
        {"quickly", 26},
        {"picks", 27},
        {"up", 28},
        {"the", 29},
        {"pace", 30},
        {"as", 31},
        {"characters", 32},
        {"become", 33},
        {"more", 34},
        {"complex", 35},
        {"and", 36},
        {"storylines", 37},
        {"engaging", 38},
        {"performances", 39},
        {"are", 40},
        {"top-notch", 41},
        {"especially", 42},
        {"by", 43},
        {"lead", 44},
        {"actors", 45},
        {"who", 46},
        {"really", 47},
        {"bring", 48},
        {"their", 49},
        {"roles", 50},
        {"to", 51},
        {"life", 52},
        {"If", 53},
        {"you", 54},
        {"enjoy", 55},
        {"gritty", 56},
        {"intense", 57},
        {"drama", 58},
        {"with", 59},
        {"a", 60},
        {"focus", 61},
        {"on", 62},
        {"moral", 63},
        {"ambiguity", 64},
        {"character", 65},
        {"development", 66},
        {"this", 67},
        {"worth", 68},
        {"watch", 69},
        {"However", 70},
        {"it", 71},
        {"is", 72},
        {"not", 73},
        {"for", 74},
        {"the", 75},
        {"faint", 76},
        {"of", 77},
        {"heart", 78},
        {"as", 79},
        {"deals", 80},
        {"with", 81},
        {"some", 82},
        {"heavy", 83},
        {"sometimes", 84},
        {"disturbing", 85},
        {"themes", 86}
    };

    // Crear el tokenizador
    Tokenizer tokenizer(vocab);

    // Texto de ejemplo
    string text = "One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. The show starts off slow, but quickly picks up the pace as the characters become more complex and the storylines more engaging. The performances are top-notch, especially by the lead actors who really bring their roles to life. If you enjoy gritty, intense drama with a focus on moral ambiguity and character development, this show is definitely worth a watch. However, it is not for the faint of heart, as it deals with some heavy and sometimes disturbing themes.";

    // Tokenizar el texto
    vector<int> token_ids = tokenizer.tokenize(text);

    // Mostrar los resultados de tokenización
    cout << "Tokens (ID's): ";
    for (int id : token_ids) {
        cout << id << " ";
    }
    cout << endl;

    // Convertir los tokens a tensor
    Tensor tensor = tokenizer.tokensToTensor(token_ids);

    // Imprimir el tensor
    cout << "Tensor de Tokens: " << endl;
    cout << tensor << endl;

    return 0;
}
