from maze_env import Maze 
from RL_brain import DeepQNetwork
import matplotlib.pyplot as plt
import numpy as np


def run_maze():
    step = 0
    EPISODES = []
    REWARDS = []
    for episode in range(20000):
        if episode == 1000:
            print(1)
        # trans = []
        # arr = []

        # initial observation
        observation = env.initialize()
        reward_sum = 0
        reward_sum_ = 0

        for epoch in range(env.max_time):
            action = RL.choose_action(observation)
            observation_, reward, _, done = env.step(action)
            reward_sum += reward
            # reward_sum_ += reward_

            if done:
                # trans.append([observation, action, reward, observation_, 0])
                # arr.append([reward_aoi,reward_e])
                RL.store_transition(observation, action, reward, observation_, 0)
            else:
                # trans.append([observation, action, reward, observation_, 1])
                # arr.append([reward_aoi, reward_e])
                RL.store_transition(observation, action, reward, observation_, 1)
            if step > 500:
                RL.learn()
            observation = observation_
            if done:
                break
            step += 1

        if episode % 50 == 0:
            reward_sum = 0
            reward_sum_ = 0

            observation = env.initialize()
            for epoch_test in range(env.max_time):
                action = RL.choose_action(observation)
                observation_, reward, _, done = env.step(action)
                reward_sum += reward
                # reward_sum_ += reward_
                if done:
                    RL.store_transition(observation, action, reward, observation_, 0)
                else:
                    RL.store_transition(observation, action, reward, observation_, 1)
                observation = observation_
                if done:
                    print(env.flag3)
                    print(env.remaining_e)
                    break

            print(episode, ' reward_sum_: ', reward_sum_)
            print(episode, ' reward_sum: ', reward_sum)
            print('aoi: ', env.aoi)
            print('uav_p: ', env.uav_p)
            EPISODES.append(episode)
            REWARDS.append(reward_sum)
    # end of game
    print('game over')
    print('uav_path: ', env.uav_path)
    print('sensor1_aoi: ',env.sensor1_aoi)
    print('sensor2_aoi: ',env.sensor2_aoi)
    print('sensor3_aoi: ',env.sensor3_aoi)
    print('sensor4_aoi: ', env.sensor4_aoi)
    print('sensor5_aoi: ', env.sensor5_aoi)
    print('sensor6_aoi: ', env.sensor6_aoi)
    print('sensor7_aoi: ', env.sensor7_aoi)
    # print('sensor8_aoi: ', env.sensor8_aoi)
    # print('sensor9_aoi: ', env.sensor9_aoi)
    print(REWARDS)
    print(EPISODES)
    print('remaining_energy:', env.remaining_e)
    return REWARDS, EPISODES
    # plt.figure()
    # plt.plot(EPISODES, REWARDS)
    # # plt.xscale('log')
    # plt.xlabel('episode')
    # plt.ylabel('reward')
    # # plt.title('sensor=1')
    # plt.show()


if __name__ == "__main__":
    # maze game
    env = Maze()
    lr = [0.02, 0.002, 0.0002]
    color = ['r', 'b', 'g']
    R = []
    E = []
    for i in range(1): # 可修改
        RL = DeepQNetwork(env.n_actions, env.n_features,
                          learning_rate=0.002,
                          reward_decay=0.9,
                          e_greedy=1,
                          replace_target_iter=300,
                          memory_size=20000,
                          # output_graph=True
                          )
        rs, es = run_maze()
        R.append(rs)
        E.append(es)
    plt.figure()
    for i in range(1):
        plt.plot(E[i], R[i],color=color[i],label='learning rate='+str(lr[i]))
    # plt.xscale('log')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    # plt.title('sensor=1')
    plt.show()