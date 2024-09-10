from collections import Counter
import nltk, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import Accuracy, Precision, Recall

nltk.download('punkt')

# Import data and labels
with open("words.json", 'r') as f1:
    words = json.load(f1)
with open("text.json", 'r') as f2:
    text = json.load(f2)
labels = np.load('labels.npy')

# Dictionaries to store the word to index mappings and vice versa
word2idx = {o:i for i,o in enumerate(words)}
idx2word = {i:o for i,o in enumerate(words)}

# Convert text to indices
for i, sentence in enumerate(text):
    text[i] = [word2idx[word] if word in word2idx else 0 for word in sentence]

# Padding function
def pad_input(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

text = pad_input(text, 50)

# Splitting dataset
train_text, test_text, train_label, test_label = train_test_split(text, labels, test_size=0.2, random_state=42)
train_data = TensorDataset(torch.from_numpy(train_text), torch.from_numpy(train_label).long())
test_data = TensorDataset(torch.from_numpy(test_text), torch.from_numpy(test_label).long())

batch_size = 400
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

# Define the classifier class
class TicketClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, target_size):
        super(TicketClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(embed_dim, target_size)

    def forward(self, text):
        embedded = self.embedding(text).permute(0, 2, 1)
        conved = F.relu(self.conv(embedded))
        conved = conved.mean(dim=2) 
        return self.fc(conved)

vocab_size = len(word2idx) + 1
target_size = len(np.unique(labels))
embedding_dim = 64

# Create an instance of the TicketClassifier class
model = TicketClassifier(vocab_size, embedding_dim, target_size)

# Training parameters
lr = 0.05
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
epochs = 3

# Train the model
model.train()
for i in range(epochs):
    running_loss, num_processed = 0,0
    for inputs, labels in train_loader:
        model.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        num_processed += len(inputs)
    print(f"Epoch: {i+1}, Loss: {running_loss/num_processed}")

# Define metrics
accuracy_metric = Accuracy(task='multiclass', num_classes=5)
precision_metric = Precision(task='multiclass', num_classes=5, average=None)
recall_metric = Recall(task='multiclass', num_classes=5, average=None)

# Evaluate model on test set
model.eval()
predicted = []

for i, (inputs, labels) in enumerate(test_loader):
    output = model(inputs)
    cat = torch.argmax(output, dim=-1)
    predicted.extend(cat.tolist())
    accuracy_metric(cat, labels)
    precision_metric(cat, labels)
    recall_metric(cat, labels)

accuracy = accuracy_metric.compute().item()
precision = precision_metric.compute().tolist()
recall = recall_metric.compute().tolist()
print('Accuracy:', accuracy)
print('Precision (per class):', precision)
print('Recall (per class):', recall)

# Define class labels
class_labels = {0:'Banking Services', 1:'Credit Card related Services',
               2:'technical issue', 3:'Fraud and Dispute', 4:'Mortgage and Loan Services'}
# Preprocess User Input
def preprocess_user_input(user_input, word2idx, seq_len=50):
    tokens = nltk.word_tokenize(user_input.lower())
    indices = [word2idx.get(token, 0) for token in tokens]
    indices = pad_input([indices], seq_len)
    return torch.tensor(indices, dtype=torch.long)

# Predict the Label
def predict_label(model, user_input, word2idx):
    input_tensor = preprocess_user_input(user_input, word2idx)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class_idx = torch.argmax(output, dim=-1).item()
    return predicted_class_idx

# Get user input and predict label
user_input = input("Enter a complaint or text for classification: ")
predicted_class_idx = predict_label(model, user_input, word2idx)
predicted_label = class_labels.get(predicted_class_idx, "Unknown")

print(f"The predicted class label is: {predicted_class_idx} ({predicted_label})")
