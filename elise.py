import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define the MLP architecture for each model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Define the overall model
class LegalAIDefender(nn.Module):
    def __init__(self, mlp1, mlp2, mlp3):
        super(LegalAIDefender, self).__init__()
        self.mlp1 = mlp1
        self.mlp2 = mlp2
        self.mlp3 = mlp3

    def forward(self, x):
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        return x

# Prepare your data
texts = [...]  # List of text data
labels = [...]  # List of corresponding labels

# Split the data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
train_features = vectorizer.fit_transform(train_texts)
val_features = vectorizer.transform(val_texts)

# Convert data to tensors
train_tensor = torch.Tensor(train_features.toarray())
val_tensor = torch.Tensor(val_features.toarray())
train_labels_tensor = torch.LongTensor(train_labels)
val_labels_tensor = torch.LongTensor(val_labels)

# Define the dimensions
input_dim = train_tensor.shape[1]
hidden_dim = 100
output_dim = 2  # assuming binary classification

# Instantiate the MLP models
mlp1 = MLP(input_dim, hidden_dim, hidden_dim)
mlp2 = MLP(hidden_dim, hidden_dim, hidden_dim)
mlp3 = MLP(hidden_dim, hidden_dim, output_dim)

# Combine the MLPs into a full model
full_model = LegalAIDefender(mlp1, mlp2, mlp3)

# Set up the training parameters
learning_rate = 0.001
batch_size = 32
epochs = 10

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(full_model.parameters(), lr=learning_rate)

# Prepare the data loaders
train_dataset = torch.utils.data.TensorDataset(train_tensor, train_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(epochs):
    full_model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()

        outputs = full_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss: {running_loss/len(train_loader):.4f}")

# Evaluation
full_model.eval()
val_outputs = full_model(val_tensor)
val_predictions = torch.argmax(val_outputs, dim=1)
val_accuracy = accuracy_score(val_labels, val_predictions.tolist())

print(f"Validation Accuracy: {val_accuracy:.4f}")

# Save the model
torch.save(full_model.state_dict(), "legal_ai_model.pth")

# Load the saved model
loaded_model = LegalAIDefender(mlp1, mlp2, mlp3)
loaded_model.load_state_dict(torch.load("legal_ai_model.pth"))
loaded_model.eval()
