import torch
import torch.nn as nn
import torch.optim as optim
from .convolutional_network import ConvNet
from .data_loader import dataset, test_dataloader, train_dataloader, validation_dataloader
from .plot_utils import plot_training_history

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)

model = ConvNet(len(dataset.classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f'*** Model ready on {device}')

epochs = 15
train_loss_values = []
validation_loss_values = []
train_accuracy_values = []
validation_accuracy_values = []

patience = 5
best_val_loss = float('inf')
epochs_no_improve = 0

print('*** Training Started')
for epoch in range(epochs):

    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        train_loss += loss.item() * batch_size
        train_total += batch_size

        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()

    train_loss_values.append(train_loss / train_total)
    train_accuracy_values.append(train_correct / train_total)

    with torch.no_grad():
        model.eval()
        validation_loss = 0.0
        val_correct = 0
        val_total = 0

        for images, labels in validation_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            valid_loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            validation_loss += valid_loss.item() * batch_size
            val_total += batch_size

            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == labels).sum().item()

        current_val_loss = validation_loss / val_total
        current_val_acc = val_correct / val_total

        validation_loss_values.append(current_val_loss)
        validation_accuracy_values.append(current_val_acc)

    print(f'Epoch {epoch+1:2d} | Train Loss: {train_loss_values[-1]:.4f} | Train Acc: {train_accuracy_values[-1]:.4f} | Val Loss: {current_val_loss:.4f} | Val Acc: {current_val_acc:.4f}')

    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_conv_model.pth')
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f'\n*** Early stopping triggered at epoch {epoch+1} ***')
        break

plot_training_history(
    train_loss_values,
    validation_loss_values,
    train_accuracy_values,
    validation_accuracy_values,
    save_path='training_history.png'
)
