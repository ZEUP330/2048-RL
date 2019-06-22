#%%
from gym_2048.env.game2048_env import Game2048Env
from deep_Q_learning.DeepQ import DeepQ
import numpy as np
import matplotlib.pyplot as plt
#%%
game = Game2048Env()
obs = game.reset()
print("state space number:", game.observation_space.shape[0])
print("type of action:", game.action_space)
# game.render()
MEMORY_CAPACITY = 30000
brain = DeepQ(batch=64, lr=0.0001, memory_capacity=MEMORY_CAPACITY,
              n_states=game.observation_space.shape[0],
              n_actions=4)
#%%
reward = []
max_block = []
for i_episode in range(5000):
    s = game.reset()
    # print(s)
    ep_r = 0
    step = 0
    last_reward = 0
    while True:
        # game.render()
        a = brain.choose_action(s)
        s_, r, done, info = game.step(np.array(a))
        if last_reward < r:
            r += max(s_)
        if 0 in np.where(np.array(s_) == max(s_))[0]:
            r += max(s_)
        r += len(np.where(np.array(s_) == 0)[0])*max(s_)
        r = int(r*0.01)

        brain.store_transition(s, a, r, s_)
        ep_r += r
        if brain.memory_counter > MEMORY_CAPACITY:
            if brain.memory_counter == MEMORY_CAPACITY+1:
                print("-------- learning --------")
            brain.learn()
        if done:
            break
        s = s_
        last_reward = r
        step += 1
    print('Ep: ', i_episode, '| Ep_r: ', ep_r,
          '|step:', step, '|max block:', max(s))
    # print(brain.memory_counter, MEMORY_CAPACITY)
    reward.append(ep_r)
    max_block.append(max(s))
brain.save_model("2048game")
plt.plot(np.linspace(0, len(reward), len(reward)), reward)
plt.show()
plt.plot(np.linspace(0, len(max_block), len(max_block)), max_block)
plt.show()
np.savetxt("reward.txt", (reward))
np.savetxt("max_block.txt", (max_block))
