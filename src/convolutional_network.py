import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ConvNet(nn.Module):
  def __init__(self, num_classes):
    super(ConvNet,self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2) 
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
    self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)

    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    self.dropout1 = nn.Dropout(p=0.25)
    self.fc1 = nn.Linear(64*12*12, 128)
    self.fc2 = nn.Linear(128, num_classes)

  def forward(self,x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.pool1(x)

    x = self.pool1(F.relu(self.conv2(x)))
    x = self.pool1(F.relu(self.conv3(x)))

    x = x.view(-1, 64*12*12)

    x = F.relu(self.fc1(x))
    x = self.dropout1(x)

    x = self.fc2(x)

    return x

class TransferNet(nn.Module):
  def __init__(self, num_classes):
    super(TransferNet,self).__init__()
    self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for parameter in self.backbone.parameters():
      parameter.requires_grad = False

    self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

  def forward(self,x):
    return self.backbone(x)