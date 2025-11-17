import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models import SimpleCNN 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on:", device)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)


model = SimpleCNN().to(device)


criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)


def train_epoch():
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


EPOCHS = 5

for epoch in range(EPOCHS):
    loss = train_epoch()
    acc = evaluate()
    print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {loss:.4f}  Accuracy: {acc:.4f}")


torch.save(model.state_dict(), "mnist_cnn.pth")
print("Model saved!")
