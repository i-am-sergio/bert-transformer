from transformers import DistilBertTokenizer, DistilBertModel
import torch

# Cargar tokenizador y modelo de DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased', output_hidden_states=True)
model.eval()

# Ejemplo con un batch de textos
texts = ["the movie is great", "this film is terrible"]
embeddings_list = []

for text in texts:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state  # Shape: [1, seq_len, 768] (DistilBERT no tiene segment embeddings)
    embeddings_list.append(embeddings)

# Guardar embeddings y labels
torch.save({"embeddings": embeddings_list, "labels": [1, 0]}, "imdb_distilbert_embeddings.pt")