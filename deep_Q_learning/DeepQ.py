from deep_Q_learning.Net import Net
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym


class DeepQ(object):
    def __init__(self, batch=32, lr=0.01, epsilon=0.9,
                 gamma=0.9, target_replay_item=100,
                 memory_capacity=2000, n_states=4, n_actions=2, env_a_shape=0):
        self.BATCH_SIZE = batch
        self.LR = lr
        self.EPSILON = epsilon
        self.GAMMA = gamma
        self.TARGET_REPLACE_ITER = target_replay_item
        self.MEMORY_CAPACITY = memory_capacity
        self.N_STATES = n_states
        self.N_ACTIONS = n_actions
        self.ENV_A_SHAPE = env_a_shape
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((self.MEMORY_CAPACITY, self.N_STATES * 2 + 2))     # initialize memory
        self.eval_net, self.target_net = Net(self.N_STATES, self.N_ACTIONS).cuda(), Net(self.N_STATES, self.N_ACTIONS).cuda()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0).reshape(-1, 1, 4, 4)
        # input only one sample
        if np.random.uniform() < self.EPSILON and self.memory_counter > self.MEMORY_CAPACITY:   # greedy
            actions_value = self.eval_net.forward(x.cuda())
            action = torch.max(actions_value, 1)[1].cpu().data.numpy()
            action = action[0] if self.ENV_A_SHAPE == 0 else action.reshape(self.ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, self.N_ACTIONS)
            action = action if self.ENV_A_SHAPE == 0 else action.reshape(self.ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.N_STATES]).reshape(-1, 1, 4, 4).cuda()
        b_a = torch.LongTensor(b_memory[:, self.N_STATES:self.N_STATES+1].astype(int)).cuda()
        b_r = torch.FloatTensor(b_memory[:, self.N_STATES+1:self.N_STATES+2]).cuda()
        b_s_ = torch.FloatTensor(b_memory[:, -self.N_STATES:]).reshape(-1, 1, 4, 4).cuda()

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't back propagate
        q_target = b_r + self.GAMMA * q_next.max(1)[0].view(self.BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, name):
        torch.save(self.eval_net, name+".pkl")
