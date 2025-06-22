#!/bin/bash
cd test
if [ "$1" == "tokenizer" ]; then
  g++ -std=c++17 testtokenizer.cpp -I../include -o testtokenizer && ./testtokenizer
elif [ "$1" == "reader" ]; then
  g++ -std=c++17 testreader.cpp -I../include -o testreader && ./testreader
elif [ "$1" == "embedding" ]; then
  g++ -std=c++17 testembedding.cpp -I../include -o testembedding && ./testembedding
elif [ "$1" == "self" ]; then
  g++ -std=c++17 testselfattention.cpp -I../include -o testselfattention && ./testselfattention
elif [ "$1" == "transformerblock" ]; then
  g++ -std=c++17 testtransformerblock.cpp -I../include -o testtransformerblock && ./testtransformerblock
else
  echo "Uso: $0 [tokenizer|reader|embedding]"
  exit 1
fi