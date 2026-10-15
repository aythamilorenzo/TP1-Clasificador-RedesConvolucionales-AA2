import torch
import torch.nn as nn
from .convolutional_network import ConvNet
from .data_loader import dataset, test_dataloader
from .plot_utils import plot_confusion_matrix

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = ConvNet(len(dataset.classes)).to(device)
criterion = nn.CrossEntropyLoss()

model_path = 'best_conv_model.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print(f'*** Loaded model weights from {model_path}')
print('*** Evaluating on TEST set...')

test_loss = 0.0
test_correct = 0
test_total = 0

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        test_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

final_test_loss = test_loss / len(test_dataloader)
final_test_acc = test_correct / test_total

print(f'\n==================================================')
print(f'FINAL TEST RESULTS:')
print(f'Test Loss: {final_test_loss:.4f}')
print(f'Test Accuracy: {final_test_acc * 100:.2f}%')
print(f'==================================================')

if final_test_acc >= 0.85:
    print("Goal achieved! Target of 85% test accuracy exceeded.")
else:
    print("Warning: Target of 85% accuracy not reached. Adjust hyperparameters.")

plot_confusion_matrix(
    y_true, 
    y_pred, 
    dataset.classes, 
    save_path='test_confusion_matrix.png'
)