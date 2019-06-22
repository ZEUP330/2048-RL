#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gym
from openai_fetch.DDPG import DDPG
import numpy as np
MAX_EPISODES = 20000
MAX_EP_STEPS = 200
MEMORY_CAPACITY = 40000

if __name__ == "__main__":
    # OrderedDict([('achieved_goal', Box(3,)), ('desired_goal', Box(3,)), ('observation', Box(10,))])
    env = gym.make("FetchReach-v1")
    rl = DDPG()
    var = 2.0
    total_rewards = []
    step_sums = []
    # 主循环
    for i in range(1, MAX_EPISODES):
        obs = env.reset()
        env.spec.max_episode_steps = 100
        state = obs.copy()
        Box_position = obs['desired_goal'].copy()
        print("Box_Position:", Box_position)

        if i % 50 == 0:
            print("\n------------------Episode:{0}------------------".format(i))
        st = 0
        rw = 0
        while True:
            st += 1
            # ------------ choose action ------------
            s = np.array(state['achieved_goal']).reshape(1, -1)
            goal = np.array(Box_position).reshape(1, -1)
            x = np.concatenate((s, goal), axis=1)
            action = [np.random.normal(rl.choose_action(x)[0], var)]

            next_state, _, done, info = env.step(np.append(action[0], [0]))
            s_next = np.array(next_state["achieved_goal"]).reshape(1, -1)
            r = -np.sqrt(np.sum(np.square(goal - s_next))).copy()
            success = info['is_success']
            x1 = np.array(np.append(next_state['achieved_goal'], next_state["desired_goal"])).reshape(1, -1)
            rl.store_transition(x, action, r, x1)
            # ----------- ---------------- -----------
            if rl.memory_counter > MEMORY_CAPACITY:
                var *= .9995
                rl.learn()
                env.render()
            rw += r
            state = next_state
            if done or st >= MAX_EP_STEPS or success:
                print("Step:{0}, total reward:{1}, average reward:{2},{3}".format(st, rw, rw*1.0/st,
                                                                                  'success' if success else '----'))
                total_rewards.append(rw)
                step_sums.append(st)
                break
