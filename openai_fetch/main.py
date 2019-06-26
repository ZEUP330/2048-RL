#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gym
from openai_fetch.DDPG import DDPG
import numpy as np
MAX_EPISODES = 20000
MAX_EP_STEPS = 200
MEMORY_CAPACITY = 100000

# ############This noise code is copied from openai baseline
# #########OrnsteinUhlenbeckActionNoise############# Openai Code#########


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


if __name__ == "__main__":
    # OrderedDict([('achieved_goal', Box(3,)), ('desired_goal', Box(3,)), ('observation', Box(10,))])
    env = gym.make("FetchReach-v1")
    rl = DDPG()
    rl.load_mode()
    var = 0.10
    rl.load_mode()
    noice = OrnsteinUhlenbeckActionNoise(mu=np.zeros(4))
    total_rewards = []
    step_sums = []
    # 主循环
    for i in range(1, MAX_EPISODES):
        obs = env.reset()
        # env.spec.max_episode_steps = MAX_EP_STEPS
        state = obs.copy()
        Box_position = obs['desired_goal'].copy()
        print("Box_Position:", Box_position)

        if i % 50 == 0:
            print("\n------------------Episode:{0}------------------".format(i))
            # rl.save_mode()
        st = 0
        rw = 0
        while True:
            # env.render()
            st += 1
            # ------------ choose action ------------
            s = np.array(state['achieved_goal']).reshape(1, -1)
            goal = np.array(Box_position).reshape(1, -1)
            x = np.concatenate((s, goal), axis=1)
            action = rl.choose_action(x)[0] + noice()

            next_state, _, done, info = env.step(action)
            s_next = np.array(next_state["achieved_goal"]).reshape(1, -1)
            r = -np.sqrt(np.sum(np.square(goal - s_next))).copy()
            success = info['is_success']
            x1 = np.array(np.append(next_state['achieved_goal'], next_state["desired_goal"])).reshape(1, -1)
            rl.store_transition(x, action, r, x1)
            # ----------- ---------------- -----------
            # if rl.memory_counter > MEMORY_CAPACITY:
            #     # print("action is :", action)
            #     # var *= .9995
            #     rl.learn()
            env.render()
            rw += r
            state = next_state
            if st >= MAX_EP_STEPS or success or done:
                print("Step:{0}, total reward:{1}, average reward:{2},{3}".format(st, rw, rw*1.0/st,
                                                                                  'success' if success else '----'))

                total_rewards.append(rw)
                step_sums.append(st)
                break
