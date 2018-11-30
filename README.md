# Value-Iteration-Network in Pytorch
![alt text](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/Value-Iteration-Network/master/resources/vin.png)

Implementation of the [Value Iteration Network](https://arxiv.org/abs/1602.02867)
## Introduction
This repo is heavily inspired by the original code (not so easy to follow) and a couple of others really well made implementation. 

- https://github.com/kentsommer/pytorch-value-iteration-networks
- https://github.com/avivt/VIN
- https://github.com/zuoxingdong/VIN_PyTorch_Visdom
## Results


| World        | Results          
| ------------- |:-------------:|
| 28x28 | ![alt text](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/Value-Iteration-Network/master/core/gridworld_28x28/figures.png)    |    

## Dataset 
The dataset is taken from [here](https://github.com/zuoxingdong/VIN_PyTorch_Visdom/tree/master/data)

For each world it contains
- the `labels`. A list of correct actions for each state
- the current state `s1`
- the target state `s2`
- the observations. A 2 channel image with 
    - in the fist channel. 1 if obstacle 0 if not
    - in the second channel. 10 if goal, 0 if not
