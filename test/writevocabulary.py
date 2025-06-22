from transformers import BertTokenizer

# Cargar el tokenizador BERT base
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Guardar el vocabulario en un archivo de texto
with open('vocab.txt', 'w') as vocab_file:
    for token in tokenizer.get_vocab():
        vocab_file.write(token + '\n')
