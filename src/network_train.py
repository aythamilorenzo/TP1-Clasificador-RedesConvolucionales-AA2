import torch
import torch.nn as nn
import torch.optim as optim

from .convolutional_network import ConvNet
from .data_loader import dataset, test_dataloader, train_dataloader, validation_dataloader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(42)

model = ConvNet(len(dataset.classes)).to(device)
criteria = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

print(f'*** Model ready')


epochs = 20

train_loss_values = []
validation_loss_values = []
train_accuracy = []
validation_accuracy = []

print('*** Train Starting')


for epoch in range(epochs):

  model.train()

  train_loss = 0.0
  for images, labels in train_dataloader:

    images, labels = images.to(device), labels.to(device)

    optimizer.zero_grad()

    outputs = model(images)
    loss = criteria(outputs, labels)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()


  train_loss_values.append(train_loss/len(train_dataloader))

  with torch.no_grad():
      model.eval()

      validation_loss = 0.0
      for images, labels in validation_dataloader:
          images, labels = images.to(device), labels.to(device)
          outputs = model(images)
          valid_loss = criteria(outputs, labels)
          validation_loss += valid_loss.item()

      validation_loss_values.append(validation_loss/len(validation_dataloader))

  print(f'Epoch {epoch+1:2d}, train loss: {train_loss_values[len(train_loss_values)-1]:.4f} validation loss: {validation_loss_values[len(validation_loss_values)-1]:.4f}')
