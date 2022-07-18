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
  * train
    * female
    * male
  * val
    * female
    * male
```

Then you can train model in distributed settings

> python train.py --path PATH 

After learning, checkpoints are stored in the 'models' file.

### Test

>python test.py --ckpt CHECKPOINT
