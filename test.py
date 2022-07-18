import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import os
import argparse

from custom_data import MyImageFolder, pil_loader

class MyTest:
  def __init__(self, model_path, img_path, transforms, device):
    self.path = img_path
    self.model_path = model_path
    self.transforms = transforms
    self.model = torchvision.models.resnet18(pretrained=False)
    self.model.fc = nn.Linear(in_features=512, out_features=2, bias=True)
    
    self.model.load_state_dict(torch.load(self.model_path))
    self.model.to(device)
    self.model.eval()
    
    self.device = device
    # self.transforms = transforms

    self.class_name = ["Female", "Male"]

  def test(self):
    
    img = pil_loader(self.path)
    
    if self.transforms:
      img = self.transforms(img)
      
    if self.device:
      img = img.to(self.device)
    
    with torch.no_grad():
      predict = self.model(img.unsqueeze_(0))
    label = torch.argmax(predict, dim=1)
    
    return img, self.class_name[label]


if __name__ == "__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
  parser = argparse.ArgumentParser()

  parser.add_argument("--ckpt", required=True, help="path of pt file of trained Resnet18 (*.pt)")
  parser.add_argument("--path", required=True, help="path of image file")

  
  
  args = parser.parse_args()
  
  transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
  
  tester = MyTest(model_path=args.ckpt, img_path = args.path, transforms=transforms, device=device)
  
  img, result = tester.test()
  # img, result = tester(os.path.abspath(args.img_file))

  img = img.squeeze_(0)
  img = img.detach().cpu().numpy().transpose((1, 2, 0))

  plt.imshow(img)
  plt.title(args.path + ", " + result)
  plt.show()
