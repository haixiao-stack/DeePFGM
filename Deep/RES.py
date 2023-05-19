import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
class BasicBlock(nn.Module):
    # 判断残差结构中，主分支的卷积核个数是否发生变化，不变则为1
    # init()：进行初始化，申明模型中各层的定义
    # downsample=None对应实线残差结构，否则为虚线残差结构
    def __init__(self, n_input,n_hidden):
        super(BasicBlock, self).__init__()
        self.hidden1 = nn.Linear(n_input,n_hidden)
        self.hidden2 = nn.Linear(n_hidden,n_input)
        # 使用ReLU作为激活函数
        self.relu = nn.ReLU()
 
    # forward()：定义前向传播过程,描述了各层之间的连接关系
    def forward(self, x):
        # 残差块保留原始输入
        identity = x
        # 如果是虚线残差结构，则进行下采样
 
        out = self.hidden1(x)
        out = self.relu(out)
        out = self.hidden2(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)
 
        return out

class ResNet(nn.Module):
    def __init__(self,block,block_num,neuron_num,n_input,n_output):
        super(ResNet, self).__init__()
        self.neuron=neuron_num
        self.input_layer=nn.Linear(n_input,self.neuron)
        self.res_layer = self._make_layer(block,block_num)
        self.output_layer=nn.Linear(self.neuron,n_output)
    def _make_layer(self, block,block_num):
        downsample = None
        # 如果满足条件，则是虚线残差结构
        layers = []
        for i in range(1, block_num):
            layers.append(block(self.neuron,self.neuron)) 
        return nn.Sequential(*layers)
    def forward(self,input):
        out = self.input_layer(input)
        out = F.relu(out)
        out = self.res_layer(out)
        out = F.relu(out)
        out = self.output_layer(out)
        return out