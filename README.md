# CelebA_classification


## Description

  This is distinguishes gender. 

## Requirements

- Torch version:1.12.0+cu113
- cuda version: 11.3
- cudnn version:8302

## Dataset 

https://www.kaggle.com/datasets/jessicali9530/celeba-dataset


## Usage

Before the train you should put dataset in train folder.<br>
example dir is :
```
* dataset
 ├── train
    ├── female
    ├── male
 ├── val
    ├── female
    ├── male
```

Then you can train model in distributed settings

> python train.py --path PATH 

* args
```
--lr              : default=0.001
--momentum        : default=0.9
--epochs          : default=25
--path            : dataset path
-- pharse         : default=['train','val']
--batch_size      : default=4
--num_workers     : default=4
```


After learning, checkpoints are stored in the 'models' file.

### Test

>python test.py --ckpt CHECKPOINT --path PATH

* args
```
--ckpt            : trained model(checkpoint) path
--path            : test image path
```
![test](https://user-images.githubusercontent.com/74218895/179892599-6de0dba4-f653-4ccc-9f63-59f7b7cfbf37.png)

