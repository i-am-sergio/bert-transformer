#include "bert.hpp"

BERT::BERT(){
    this->message = "========== BERT esta vivo! =========="; 
}

void BERT::printMessage() {
    std::cout << message << std::endl;
}
