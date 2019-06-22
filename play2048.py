from gym_2048.env.game2048_env import Game2048Env
from baselines import deepq
from Env.env2048 import core


def main():
    env = core.Env("C:/Users/VCC/Desktop/project/2048-python/Env/img/")
    # env = Game2048Env()
    # act = deepq.learn(env, network='mlp', total_timesteps=0, load_path="game2048.pkl")
    #
    env.reset()
    while True:
        env.render()
    #     obs, done = env.reset(), False
    #     episode_rew = 0
    #     while not done:
    #         # env.render()
    #         obs, rew, done, _ = env.step(act(obs[None])[0])
    #         episode_rew += rew
    #     print("Episode reward", episode_rew)
    #     print(max(obs))


if __name__ == '__main__':
    main()
