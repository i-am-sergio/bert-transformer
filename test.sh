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
elif [ "$1" == "encoder" ]; then
  g++ -std=c++17 test_transformer_encoder.cpp -I../include -o test_transformer_encoder && ./test_transformer_encoder
elif [ "$1" == "test" ]; then
  g++ -std=c++17 test.cpp -I../model -o test && ./test
else
  echo "Uso: $0 [tokenizer|reader|embedding|posencode|mha|encoder|bert]"
  exit 1
fi