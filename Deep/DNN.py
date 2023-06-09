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
    def __init__(self, n_input,n_output):
        super(BasicBlock, self).__init__()
        self.hidden = nn.Linear(n_input,n_output)
        self.relu = nn.ReLU()
 
    # forward()：定义前向传播过程,描述了各层之间的连接关系
    def forward(self, x):
        # 残差块保留原始输入
        
        out = self.hidden(x)
        out = self.relu(out)

 
        return out

class DeepNet(nn.Module):
    def __init__(self,block,block_num,neuron_num,n_input,n_output):
        super(ResNet, self).__init__()
        self.neuron=neuron_num
        self.input_layer=nn.Linear(n_input,self.neuron)
        self.dnn_layer = self._make_layer(block,block_num)
        self.output_layer=nn.Linear(self.neuron,n_output)
    def _make_layer(self, block,block_num):
        # 如果满足条件，则是虚线残差结构
        layers = []
        for i in range(1, block_num):
            layers.append(block(self.neuron,self.neuron)) 
        return nn.Sequential(*layers)
    def forward(self,input):
        out = self.input_layer(input)
        out = F.relu(out)
        out = self.dnn_layer(out)
        out = F.relu(out)
        out = self.output_layer(out)
        return out