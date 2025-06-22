#!/bin/bash
cd test
if [ "$1" == "tokenizer" ]; then
  g++ -std=c++17 test_tokenizer.cpp -I../include -o test_tokenizer && ./test_tokenizer
elif [ "$1" == "reader" ]; then
  g++ -std=c++17 testreader.cpp -I../include -o testreader && ./testreader
elif [ "$1" == "embedding" ]; then
  g++ -std=c++17 test_embedding.cpp -I../include -o test_embedding && ./test_embedding
elif [ "$1" == "posencode" ]; then
  g++ -std=c++17 test_positionalencode.cpp -I../include -o test_positionalencode && ./test_positionalencode
elif [ "$1" == "mha" ]; then
  g++ -std=c++17 test_multiheadattention.cpp -I../include -o test_multiheadattention && ./test_multiheadattention
elif [ "$1" == "self" ]; then
  g++ -std=c++17 testselfattention.cpp -I../include -o testselfattention && ./testselfattention
elif [ "$1" == "transformerblock" ]; then
  g++ -std=c++17 testtransformerblock.cpp -I../include -o testtransformerblock && ./testtransformerblock
else
  echo "Uso: $0 [tokenizer|reader|embedding]"
  exit 1
fi