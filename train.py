import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from legal_ai_model import LegalAIDefender  # Assuming your model class is defined in legal_ai_model.py
from dataset import LegalAIDataset  # Assuming you have a custom dataset class defined in dataset.py

# Set the device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the random seed for reproducibility
torch.manual_seed(42)

# Define the training function
def train(model, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)

# Define the evaluation function
def evaluate(model, val_loader):
    model.eval()
    val_predictions = []
    val_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            _, predictions = torch.max(outputs, dim=1)

            val_predictions.extend(predictions.tolist())
            val_labels.extend(labels.tolist())

    val_accuracy = accuracy_score(val_labels, val_predictions)
    return val_accuracy

def main():
    # Set up the training parameters
    learning_rate = 0.001
    batch_size = 32
    epochs = 10

    # Load the dataset
    dataset = LegalAIDataset()  # Replace with your actual dataset class and parameters
    train_dataset, val_dataset = dataset.get_train_val_datasets()

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate the model
    model = LegalAIDefender()  # Replace with your actual model instantiation

    # Move the model to the device
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion)
        val_accuracy = evaluate(model, val_loader)

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "legal_ai_model.pth")

if __name__ == "__main__":
    main()
