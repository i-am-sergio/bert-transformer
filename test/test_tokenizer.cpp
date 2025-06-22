#include "Tokenizer.hpp"

int main() {
    // Cargar vocabulario
    Tokenizer tokenizer("vocab.txt");

    // Tokenizar una frase
    string sentence = "I love this movie!";
    vector<string> tokens = tokenizer.tokenize(sentence);
    tokenizer.print_tokens(tokens);  // Imprime los tokens

    // Convertir los tokens a IDs
    Tensor token_ids = tokenizer.convert_tokens_to_ids(tokens);
    cout << "Tokens convertidos a IDs: " << token_ids << endl;

    return 0;
}
