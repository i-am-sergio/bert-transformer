"""
1. Cargar y Preprocesar el Dataset IMDB
"""
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

# Cargar el dataset (asegúrate de que esté en el mismo directorio)
df = pd.read_csv("../datasets/IMDB_Dataset.csv")  # Asume columnas: "review" (texto) y "sentiment" ("positive"/"negative")
df["label"] = df["sentiment"].apply(lambda x: 1 if x == "positive" else 0)  # Convertir a 1/0

# Tomar un subconjunto para prueba rápida (opcional)
df = df.sample(1000)  # Comienza con 1000 muestras para debug, luego usa las 50,000

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
model.eval()

# Clase Dataset personalizada
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_length
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long)
        }

# Crear Dataset y DataLoader
dataset = IMDBDataset(df["review"].tolist(), df["label"].tolist(), tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)


"""
2. Extraer Embeddings con DistilBERT y Guardarlos
"""
def extract_embeddings(dataloader, model):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extrayendo embeddings"):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings.append(outputs.last_hidden_state[:, 0, :])  # Usar el embedding [CLS]
            labels.append(batch["label"])
    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)

# Extraer embeddings (esto puede tardar varias horas para 50,000 datos, usa GPU si es posible)
embeddings, labels = extract_embeddings(dataloader, model)

# Guardar embeddings y labels
torch.save({"embeddings": embeddings, "labels": labels}, "imdb_distilbert_embeddings.pt")

"""
3. Implementación del Modelo de Clasificación
"""
import torch.nn as nn

class SentimentClassifier(nn.Module):
    def __init__(self, input_dim=768, num_classes=1):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(x))

# Cargar embeddings
data = torch.load("imdb_distilbert_embeddings.pt")
X, y = data["embeddings"], data["labels"]

# Dividir en train/test (80/20)
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

"""
4. Entrenamiento y Evaluación
"""
# Configuración
model = SentimentClassifier()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Bucle de entrenamiento
epochs = 10
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train).squeeze()
    loss = criterion(outputs, y_train.float())
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# Evaluación
model.eval()
with torch.no_grad():
    y_pred = model(X_test).squeeze()
    y_pred_class = (y_pred > 0.5).float()
    accuracy = (y_pred_class == y_test).float().mean()
    print(f"Accuracy en test: {accuracy.item() * 100:.2f}%")


"""
5. Predicción en una Nueva Reseña
"""
def predict_sentiment(text, model, tokenizer, max_length=512):
    model.eval()
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        padding="max_length", 
        truncation=True, 
        max_length=max_length
    )
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state[:, 0, :]  # Embedding [CLS]
        prediction = model(embeddings).item()
    return "Positive" if prediction > 0.5 else "Negative"

# Ejemplo de predicción
new_review = "This movie was a waste of time. Terrible acting and plot."
print(f"Predicción: {predict_sentiment(new_review, model, tokenizer)}")
# Output: Predicción: Negative