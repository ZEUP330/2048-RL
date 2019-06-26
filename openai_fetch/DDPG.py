#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torchvision.models as models
import numpy as np
import os
import math
import random
import time


#####################  hyper parameters  ####################

LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.99     # reward discount
TAU = 0.001      # soft replacement
MEMORY_CAPACITY = 100000
BATCH_SIZE = 128
N_STATES = 3
RENDER = False
EPSILON = 0.9
ENV_PARAMS = {'obs': 3, 'goal': 3, 'action': 4, 'action_max': 0.2}
# ##############################  DDPG  ####################################


class ANet(nn.Module):   # ae(s)=a
    def __init__(self):
        super(ANet, self).__init__()
        self.max_action = ENV_PARAMS['action_max']
        self.state = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Linear(ENV_PARAMS['obs'] + ENV_PARAMS['goal'], 400)),
                ('relu1', nn.ReLU()),
                ('conv2', nn.Linear(400, 300)),
                ('relu2', nn.ReLU()),
                ('conv3', nn.Linear(300, ENV_PARAMS['action'])),
                # ('tanh1', nn.Tanh())
            ]))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, s):
        return self.state(s)


class CNet(nn.Module):
    def __init__(self):
        super(CNet, self).__init__()
        self.state = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Linear(ENV_PARAMS['obs'] + ENV_PARAMS['goal'] + ENV_PARAMS['action'], 400)),
                ('relu1', nn.ReLU()),
                ('conv2', nn.Linear(400, 300)),
                ('relu2', nn.ReLU()),
                ('conv3', nn.Linear(300, 1))
            ]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, s, a):
        x = torch.cat((s, a), dim=1)
        return self.state(x)


class DDPG(object):
    def __init__(self):
        self.memory = np.zeros((MEMORY_CAPACITY, (ENV_PARAMS['obs'] + ENV_PARAMS['goal'])*2 + ENV_PARAMS['action'] + 1))
        self.memory_counter = 0  # 记忆库计数
        self.Actor_eval = ANet().cuda()
        self.Actor_target = ANet().cuda()
        self.Critic_eval = CNet().cuda()
        self.Critic_target = CNet().cuda()
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(), lr=LR_C)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(), lr=LR_A)
        self.loss_td = nn.MSELoss()
        self.L1Loss = nn.SmoothL1Loss()
        self.f = 0

    def choose_action(self, s):
        state = torch.FloatTensor(s).cuda()
        return self.Actor_eval(state).cpu().data.numpy()

    def learn(self):
        self.f += 1
        self.f %= 100
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s_num = ENV_PARAMS['obs'] + ENV_PARAMS['goal']
        b_a_num = ENV_PARAMS['action']
        b_s = torch.FloatTensor((b_memory[:, :b_s_num]).reshape(-1, b_s_num)).cuda()
        b_a = torch.FloatTensor((b_memory[:, b_s_num:b_s_num + b_a_num]).reshape(-1, b_a_num)).cuda()
        b_r = torch.FloatTensor((b_memory[:, b_s_num + b_a_num:b_s_num + b_a_num + 1]).reshape(-1, 1)).cuda()
        b_s_ = torch.FloatTensor((b_memory[:, b_s_num + b_a_num + 1:2*b_s_num+b_a_num + 1]).reshape(-1, b_s_num)).cuda()

        # Compute the target Q value
        target_Q = self.Critic_target(b_s_, self.Actor_target(b_s_))
        target_Q = b_r + (GAMMA * target_Q).detach()

        # Get current Q estimate
        current_Q = self.Critic_eval(b_s, b_a)

        # Compute critic loss
        critic_loss = self.loss_td(current_Q, target_Q)

        # Optimize the critic
        self.ctrain.zero_grad()
        critic_loss.backward()
        self.ctrain.step()

        # Compute actor loss
        actor_loss = self.L1Loss(self.Critic_eval(b_s, self.Actor_eval(b_s)).mean(), torch.zeros(()).cuda())
        if self.f == 0:
            print(actor_loss)
        # Optimize the actor
        self.atrain.zero_grad()
        actor_loss.backward()
        self.atrain.step()

        self.soft_update(self.Actor_target, self.Actor_eval, TAU)
        self.soft_update(self.Critic_target, self.Critic_eval, TAU)

    def store_transition(self, s, a, r, s_):

        s_num = ENV_PARAMS['obs'] + ENV_PARAMS['goal']
        a_num = ENV_PARAMS['action']
        s = np.array(s).reshape(-1, s_num)
        a = np.array(a).reshape(-1, a_num)
        r = np.array(r).reshape(-1, 1)
        s_ = np.array(s_).reshape(-1, s_num)
        transition = np.hstack((s, a, r, s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data*(1.0 - tau) + param.data*tau
            )

    def save_mode(self):
        torch.save(self.Actor_eval, "Actor_eval.pkl")
        torch.save(self.Actor_target, "Actor_target.pkl")
        torch.save(self.Critic_eval, "Critic_eval.pkl")
        torch.save(self.Critic_target, "Critic_target.pkl")

    def load_mode(self):
        self.Actor_eval = torch.load("Actor_eval.pkl")
        self.Actor_target = torch.load("Actor_target.pkl")
        self.Critic_eval = torch.load("Critic_eval.pkl")
        self.Critic_target = torch.load("Critic_target.pkl")
