import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, path, transforms=None):
        self.path = path
        self.transforms = transforms
        self.img_path = os.listdir(self.path)
        
        
    def __len__(self):
        img, class_idx, length = make_idx(self.img_path, self.path)
        return length

    def __getitem__(self, index):
        img, class_idx, length = make_idx(self.img_path, self.path)
        class_idx = np.array(class_idx)
        class_idx = class_idx.astype('long')

        if self.transforms is not None:
            img = self.transforms(img)

        return img, class_idx


def GetDataSize(mypath):
    dir_phase = os.listdir(mypath)
    data_size=0

    for phase in dir_phase:
        file_dir = os.listdir(os.path.join(mypath, phase))
        data_size += len(file_dir)

    return data_size


def make_idx(myclass, path):
    class_idx = []
    file_list = []
    length = 0
    for i, j in enumerate(myclass):
        class_idx.append(i)
        m_p = path + '/' + j
        file_list = os.listdir(m_p)
        length += (len(file_list))
        
    for i, j in enumerate(myclass):
        m_p = path + '/' + j
        for file in file_list:
            _path = os.path.join(m_p,file) 
            if os.path.isfile(_path):
                with open(_path, 'rb') as f:
                    with Image.open(f) as img:
                        my_img = img.convert('RGB')

    return my_img, class_idx, length

def MyDataLoader(args):
  data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
  }

  train_path = os.path.join(args.path, 'train')
  val_path = os.path.join(args.path, 'val')
  for pharse in args.pharse:
    if pharse == 'train':
      my_dset = MyImageFolder(
          train_path,
          transforms=data_transforms['train']
      )
      train_loader = DataLoader(
          my_dset, batch_size = args.batch_size,
          shuffle = True, num_workers = args.num_workers
      )

    else:
      my_dset = MyImageFolder(
          val_path,
          transforms=data_transforms['val']
      )
      val_loader = DataLoader(
          my_dset, batch_size = args.batch_size,
          shuffle = True, num_workers = args.num_workers
      )

  return train_loader, val_loader
  
  
def GetDataDir(mypath):
  my_class = os.listdir(mypath)
  class_idx = {}
  length = 0
  file_dir = []
  imgs = []

  for i, j in enumerate(my_class):
    class_idx[j] = i

    file_list = os.listdir(mypath+'/'+j)
    length += (len(file_list))

    for f in file_list:
      path = mypath+'/'+j+'/'+f
      file_dir.append(path)

      imgs.append((path, i))

  return my_class, class_idx, length, file_dir, imgs
  
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')  

class MyImageFolder(Dataset):
  def __init__(self, root, transforms=None, target_transform=None, loader=pil_loader):
    classes, class_to_idx, length, img_path, imgs = GetDataDir(root)
    
    if len(imgs) == 0:
        raise(RuntimeError("Found 0 images in subfolders of: " + root))

    self.root = root
    self.imgs = imgs
    self.classes = classes
    self.class_to_idx = class_to_idx
    self.transforms = transforms
    self.target_transform = target_transform
    self.img_path = img_path
    self.loader = loader

  def __getitem__(self, index):
      path, target = self.imgs[index]
      img = self.loader(path)
      if self.transforms is not None:
          img = self.transforms(img)
      if self.target_transform is not None:
          target = self.target_transform(target)

      return img, target


  def __len__(self):
      return len(self.imgs)

      