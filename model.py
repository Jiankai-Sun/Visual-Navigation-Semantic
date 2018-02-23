from __future__ import division
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from utils import norm_col_init, weights_init, get_upsampling_weight


class A3Clstm(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(A3Clstm, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        # self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        # self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)
        # self.bn4 = nn.BatchNorm2d(64)

        # self.lstm = nn.LSTMCell(1024, 512)
        self.lstm = nn.LSTMCell(2249, 512)
        num_outputs = action_space
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_outputs)

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)
        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        # For Fully Convolutional Layers
        self.upscore1 = nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, output_padding=1, bias=False)
        self.upscore1.weight.data.normal_(0.0, 0.02)
        self.upscore2 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, output_padding=1, bias=False)
        self.upscore2.weight.data.copy_(get_upsampling_weight(1, 1, 4))
        self.upscore3 = nn.ConvTranspose2d(1, 1, kernel_size=5, stride=2, bias=False)
        self.upscore3.weight.data.copy_(get_upsampling_weight(1, 1, 5))
        self.upscore4 = nn.ConvTranspose2d(1, 1, kernel_size=5, stride=2, bias=False)
        self.upscore4.weight.data.copy_(get_upsampling_weight(1, 1, 5))

        self.train()

    def forward(self, inputs):
        input1, (hx, cx), input2 = inputs
        x1 = self.maxp1(self.conv1(input1))
        # x1 = self.bn1(F.relu(x1))
        x2 = self.maxp2(self.conv2(x1))
        # x2 = self.bn2(F.relu(x2))
        x3 = self.maxp3(self.conv3(x2))
        # x3 = self.bn3(F.relu(x3))
        x4 = self.maxp4(self.conv4(x3))
        # x4 = self.bn4(F.relu(x4))

        x5 = x4.view(x4.size(0), -1)
        input2 = input2.view(-1, input2.size(0))

        x6 = torch.cat((x5, input2), 1)

        hx, cx = self.lstm(x6, (hx, cx))

        # For Fully Convolutional Layers
        upscore1 = self.upscore1(x4)  # ([1, 1, 9, 13])
        upscore1 = upscore1[:, :, 1:-1, 1:-1]
        upscore2 = self.upscore2(upscore1) # ([1, 1, 18, 26])
        upscore2 = upscore2[:, :, 1:-1, 1:-1]
        upscore3 = self.upscore3(upscore2)  # (9)
        upscore4 = self.upscore4(upscore3)  # ([1, 1, 105, 137])
        upscore4 = upscore4[:, :, :-3, :-5].contiguous()
        # print("x1.size(): ", x1.size())
        # print("x2.size(): ", x2.size())
        # print("x3.size(): ", x3.size())
        # print("x4.size(): ", x4.size())
        #
        # print("upscore1.size(): ", upscore1.size())
        # print("upscore2.size(): ", upscore2.size())
        # print("upscore3.size(): ", upscore3.size())
        # print("upscore4.size(): ", upscore4.size())
        #
        # x1.size():  torch.Size([1, 32, 45, 60])
        # x2.size():  torch.Size([1, 32, 21, 29])
        # x3.size():  torch.Size([1, 64, 10, 14])
        # x4.size():  torch.Size([1, 64, 5, 7])
        # upscore1.size():  torch.Size([1, 1, 10, 14])
        # upscore2.size():  torch.Size([1, 1, 21, 29])
        # upscore3.size():  torch.Size([1, 1, 45, 61])
        # upscore4.size():  torch.Size([1, 1, 90, 120])

        return self.critic_linear(hx), self.actor_linear(hx), (hx, cx), upscore4
