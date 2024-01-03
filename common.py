#!/usr/bin/env python

import os

import torch
import torch.nn as nn


class ToyModel(nn.Module):
  def __init__(self):
    super(ToyModel, self).__init__()
    self.net1 = nn.Linear(10, 10)
    self.relu = nn.ReLU()
    self.net2 = nn.Linear(10, 5)

  def forward(self, x):
    return self.net2(self.relu(self.net1(x)))


class ToyMpModel(nn.Module):
  def __init__(self, *devices):
    super(ToyMpModel, self).__init__()
    self.devices = devices
    self.net0 = torch.nn.Linear(10, 10).to(devices[0])
    self.net1 = torch.nn.Linear(10, 10).to(devices[1])
    self.net2 = torch.nn.Linear(10, 10).to(devices[2])
    self.net3 = torch.nn.Linear(10, 5).to(devices[3])
    self.relu = torch.nn.ReLU()

  def forward(self, x):
    x = self.relu(self.net0(x.to(self.devices[0])))
    x = self.relu(self.net1(x.to(self.devices[1])))
    x = self.relu(self.net2(x.to(self.devices[2])))
    return self.net3(x.to(self.devices[3]))
