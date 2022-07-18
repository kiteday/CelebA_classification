from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import argparse
import os

from custom_data import MyDataLoader, GetDataSize

class MyTrain():
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  def __init__(self, path, epochs, lr, momentum, args, device):
    self.path=path
    self.model = models.resnet18(pretrained=True)
    self.model.fc = nn.Linear(self.model.fc.in_features, 2)
    self.model_ft = self.model.to(device)
    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = optim.SGD(self.model_ft.parameters(), lr=lr, momentum=momentum)
    self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
    self.epochs = epochs
    self.args = args
    self.device = device

  def train_model(self):
    # since = time.time
    best_model_wts = copy.deepcopy(self.model_ft.state_dict())
    best_acc = 0.0
    running_loss = 0.0
    running_corrects =0.0
    
    
    for epoch in range(0, self.epochs):
      

      self.model_ft.train()
      running_loss = 0.0
      running_corrects = 0

      data_loader, _ = MyDataLoader(self.args)
      for inputs, labels in data_loader:
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(True):
          outputs = self.model_ft(inputs).float()
          _, preds = torch.max(outputs, 1)
          loss = self.criterion(outputs, labels)

          # backward + optimize only if in training phase
          loss.backward()
          self.optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
        self.scheduler.step()
      
      dataset_sizes = GetDataSize(os.path.join(self.path, 'train'))

      epoch_loss = running_loss / dataset_sizes
      epoch_acc = running_corrects.double() / dataset_sizes
      
      print(f'Epoch {epoch}/{self.epochs - 1}')
      
      print('-' * 10)
      print(f' Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
  
      val_acc = self.val()
      if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(self.model_ft.state_dict())
        torch.save(self.model_ft.state_dict(), f"./models/model({epoch}).pt")
  
    # time_elapsed = time.time() - since
    # print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    # print(f'Best val Acc: {best_acc:4f}')

    self.model_ft.load_state_dict(best_model_wts)
    return self.model_ft

  def val(self):
    # since = time.time
    best_model_wts = copy.deepcopy(self.model_ft.state_dict())
    # best_acc = 0.0

    self.model_ft.eval()
    running_loss = 0.0
    running_corrects = 0

    _, data_loader = MyDataLoader(self.args)
    for inputs, labels in data_loader:
      inputs = inputs.to(self.device)
      labels = labels.to(self.device)

      # zero the parameter gradients
      self.optimizer.zero_grad()
      
      with torch.set_grad_enabled(False):
        outputs = self.model_ft(inputs).float()
        _, preds = torch.max(outputs, 1)
        loss = self.criterion(outputs, labels)
        
      running_loss += loss.item() * inputs.size(0)
      running_corrects += torch.sum(preds == labels.data)


    dataset_sizes = GetDataSize(os.path.join(self.path, 'val'))

    epoch_loss = running_loss / dataset_sizes
    epoch_acc = (running_corrects) / dataset_sizes
    
    print(f' Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # if best_acc < epoch_acc:
    #   best_acc = epoch_acc

    return epoch_acc


if __name__ == "__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
  parser.add_argument('--momentum', type=float, default=0.9)
  parser.add_argument('--epochs', type=int, default=25)
  parser.add_argument('--path', type=str, help="dataset path")
  parser.add_argument('--pharse', type=list, default=['train', 'val'])
  parser.add_argument('--batch_size', type=int, default=4)
  parser.add_argument('--num_workers', type=int, default=4)

  args = parser.parse_args()

  my_train = MyTrain(path=args.path, epochs=args.epochs, lr=args.lr, momentum=args.momentum, args=args, device=device)
  my_train.train_model()
