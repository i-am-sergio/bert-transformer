#pragma once
#include <iostream>

class BERT {
private:
    std::string message;
public:
    BERT(){
        this->message = "========== BERT esta vivo! =========="; 
    }
    // Otras declaraciones de métodos (si los hay)
    void printMessage(){
        std::cout << message << std::endl;
    }
};