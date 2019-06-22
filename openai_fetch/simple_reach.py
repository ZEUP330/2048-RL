#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gym
import time
MAX_EPISODES = 20000
MAX_EP_STEPS = 200
MEMORY_CAPACITY = 10000

if __name__ == "__main__":
    # OrderedDict([('achieved_goal', Box(3,)), ('desired_goal', Box(3,)), ('observation', Box(10,))])
    env = gym.make("FetchReach-v1")
    var = 3.0
    total_rewards = []
    step_sums = []
    # 主循环
    obs = env.reset()
    for i in range(1, MAX_EPISODES):
        env.render()
        time.sleep(0.1)
        state = obs['achieved_goal'].copy()
        Box_position = obs['desired_goal'].copy()
        obs = obs['observation'].copy()
        print("target_Position:", Box_position)
        print("ender_Position:", state)
        action = [0, 0, i/MAX_EP_STEPS, 0]  # [向前 ,向左 ,向上 ,未知 ]
        obs, r, info, next_state = env.step(action)

