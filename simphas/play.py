#!/usr/bin/env python3
import logging
import click
import numpy as np
from os.path import abspath, dirname, join
from gym.spaces import Tuple
from mujoco_py import const, MjViewer
from mae_envs.viewer.env_viewer import EnvViewer
from mae_envs.wrappers.multi_agent import JoinMultiAgentActions
from mujoco_worldgen.util.envs import examine_env, load_env
from mujoco_worldgen.util.types import extract_matching_arguments
from mujoco_worldgen.util.parse_arguments import parse_arguments
from runpy import run_path
from mae_envs.modules.util import (uniform_placement, center_placement,
                                   uniform_placement_middle)

from gym.spaces import Box, MultiDiscrete, Discrete

from simphas.MRL import mpolicy


import gym
from RL_brain_2 import PolicyGradient
import matplotlib.pyplot as plt

def edge_punish(x,y,l=1,p=3.6,w=5.0):
    if (np.abs(x-0)<l) | (np.abs(x-p)<l) | (np.abs(y-0)<l) | (np.abs(y-p)<l):
        return w
    else:
        return 0.0

def matdis(n, obs_x):
    dism = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dism[i, j] = np.sqrt(np.sum((obs_x[i, :2] - obs_x[j, :2])**2))
    return dism
def matmas(n,mas):
    matm = np.empty([n,n],dtype= bool)
    for i in range(n):
        for j in range(n):
            if i >  j:
                matm[i, j] = mas[i,j]
            elif i < j:
                matm[i, j] = mas[i, j-1]
            else:
                matm[i, j] = False
    return matm


def game_rew(n,n_seekers, dism, matm, thr=1.0):
    return np.sum( (np.sum ( ((dism < np.ones((n,n))*thr) & (matm))[-n_seekers:], axis=0)>0))



def main():
    kwargs = {}
    env_name = 'mae_envs/envs/mybase.py'

    display = True
    n_agents= 2
    n_seekers=1
    n_hiders=1
    episode=500
    n_episode=300
    kwargs.update({
        'n_agents': n_agents,
        'n_seekers': n_seekers,
        'n_hiders': n_hiders,
        'n_boxes':1,
        'cone_angle': 2 * np.pi,
        #'n_substeps' : 1

    })

    module = run_path(env_name)
    make_env = module["make_env"]
    args_to_pass, args_remaining = extract_matching_arguments(make_env, kwargs)
    env = make_env(**args_to_pass)
    env.reset()
    env_viewer = EnvViewer(env)


    RL = mpolicy(
        n_actions=9,
        n_features=8,
        #n_features=4,
        learning_rate=0.01,
        reward_decay=0.9,
        units=30
        # output_graph=True,
    )

    RRL = PolicyGradient(
        n_actions=9,
        n_features=8,
        learning_rate=0.01,
        reward_decay=0.9,
        # output_graph=True,
    )

    a=[]
    speed=3
    for ii in range(n_episode):
        env_viewer.env_reset()
        sampleaction = np.array([[5, 5, 5], [5, 5, 5]])
        action = {'action_movement': sampleaction}


        obs, rew, down, _ = env_viewer.step(action)
        observation = np.array([obs['observation_self'][1][0], obs['observation_self'][1][1], obs['observation_self'][1][4],obs['observation_self'][1][5], obs['observation_self'][0][0], obs['observation_self'][0][1],obs['observation_self'][0][4],obs['observation_self'][0][5]])
        #observation = np.array([obs['observation_self'][1][0], obs['observation_self'][1][1],obs['observation_self'][1][4],obs['observation_self'][1][5]])

        for i in range(episode):
            action = RL.action(observation, speed=speed)
            actionrrl = RRL.choose_action(observation)

            h1=actionrrl//3+4
            h2=actionrrl%3+4
            #print(action)
            a1=action[1][0]
            a2=action[1][1]
            ac = {'action_movement': np.array([[h1, h2, 5], [a1, a2, 5]])}
            #print(ac)
            obs_, reward, done, info = env_viewer.step(ac, show=True)
            observation_ = np.array([obs_['observation_self'][1][0], obs_['observation_self'][1][1], obs_['observation_self'][1][4], obs_['observation_self'][1][5],obs_['observation_self'][0][0], obs_['observation_self'][0][1],obs_['observation_self'][0][4], obs_['observation_self'][0][5]])
            #observation_ = np.array(
            #    [obs_['observation_self'][1][0], obs_['observation_self'][1][1], obs_['observation_self'][1][4],
            #      obs_['observation_self'][1][5]])

            #rew = 1-np.sqrt((observation_[0] - observation_[4]) ** 2 + (observation_[1] - observation_[5]) ** 2)
            #rew = 1 - np.sqrt((observation_[0] - observation_[4]) ** 2 + (observation_[1] - observation_[5]) ** 2)

            if not obs_['mask_aa_obs'][1][0]:
                rew=-( np.sqrt((observation_[0] - observation_[4]) ** 2 + (observation_[1] - observation_[5]) ** 2)*5)
            else:
                rew = -( np.sqrt((observation_[0] - observation_[4]) ** 2 + (observation_[1] - observation_[5]) ** 2)*5)
            #obs['observation_self'][0][0] = 1
            #obs['observation_self'][0][1] = 1
            #dism = matdis(n_agents, obs_['observation_self'])
            #matm = matmas(n_agents, obs_['mask_aa_obs'])
            #rew = game_rew(n_agents, n_seekers, dism, matm, thr=1.5)
            #if rew>0.0:
            #    rew=1.0
            #else:
            #    rew-1.0
            #print(rew)
            '''
            obs = np.array([obs['observation_self'][0, :2], obs['observation_self'][1, :2]])
            #print(obs)
            ac = RL.action(obs.reshape(-1))
            action = {'action_movement': ac}
            obs, rew, down, _ = env_viewer.step(action,show=True)

            #obs['observation_self'][0][0] = 1
            #obs['observation_self'][0][1] = 1




            #dism = matdis(n_agents, obs['observation_self'])
            #matm = matmas(n_agents, obs['mask_aa_obs'])



            #rew = game_rew(n_agents, n_seekers, dism, matm, thr=1)
            '''
            #RRL.store_transition(observation, actionrrl, -rew-edge_punish(observation_[4],observation_[5]))
            RRL.store_transition(observation, actionrrl, -rew)
            #RL.get_rew(rew-edge_punish(observation_[1],observation_[0]))
            RL.get_rew(rew)

            #print(50-edge_punish(observation_[0],observation_[1]))
            observation = observation_

        if ii>(n_episode-51):
            speed=1
            RL.learning_rate=0
            RRL.learning_rate = 0
            a.append(RRL.ep_rs)
        else:
            print(ii)
        RL.learn()
        RRL.learn()












        np.save('hahd.npy', a)
        #print(ii,R)
'''
def test():
    kwargs = {}
    env_name = 'mae_envs/envs/mybase.py'

    display = True
    n_agents = 2
    n_seekers = 1
    n_hiders = 1
    episode = 200
    n_episode = 1000
    kwargs.update({
        'n_agents': n_agents,
        'n_seekers': n_seekers,
        'n_hiders': n_hiders,
        'n_boxes': 0,
        'cone_angle': 2 * np.pi
    })

    module = run_path(env_name)
    make_env = module["make_env"]
    args_to_pass, args_remaining = extract_matching_arguments(make_env, kwargs)
    env = make_env(**args_to_pass)
    env.reset()
    env_viewer = EnvViewer(env)




    RL = PolicyGradient(
        n_actions=9,
        n_features=4,
        learning_rate=0.01,
        reward_decay=0.99,
        # output_graph=True,
    )

  

    for i_episode in range(3000):

        env_viewer.env_reset()

        sampleaction = np.array([[5, 5, 5], [5, 5, 5]])
        action = {'action_movement': sampleaction}

        obs, rew, down, _ = env_viewer.step(action)
        observation = np.array([obs['observation_self'][1][0], obs['observation_self'][1][1], obs['observation_self'][1][4], obs['observation_self'][1][5]])
        #observation = np.array([obs['observation_self'][0][0], obs['observation_self'][0][1], obs['observation_self'][0][4], obs['observation_self'][0][5], obs['observation_self'][1][0], obs['observation_self'][1][1], obs['observation_self'][1][4], obs['observation_self'][1][5]])

        for ii in range(1000):
            action,BL = RL.choose_action(observation)

            a1=action//3+4
            a2=action%3+4
            ac = {'action_movement': np.array([[5,5,5],[a1,a2,5]])}
            obs_, reward, done, info = env_viewer.step(ac, show=True)
            observation_=np.array([obs_['observation_self'][1][0], obs_['observation_self'][1][1],obs['observation_self'][1][4],obs['observation_self'][1][5]])
            #observation_ = np.array(
            #    [obs_['observation_self'][0][0], obs_['observation_self'][0][1], obs_['observation_self'][0][4],
             #    obs_['observation_self'][0][5], obs_['observation_self'][1][0], obs_['observation_self'][1][1],
            #     obs_['observation_self'][1][4], obs_['observation_self'][1][5]])

            #reward =(2-np.sqrt((observation_[0]-1)**2+(observation_[1]-1)**2))
            #reward = (2 - np.sqrt((observation_[0] - observation_[4]) ** 2 + (observation_[1] - observation_[5]) ** 2))
            reward = (2 - np.sqrt((1 - observation_[0]) ** 2 + (1 - observation_[1]) ** 2))
            #print(reward)

            #print(reward,observation,action)
            RL.store_transition(observation, action, reward,BL)

            observation = observation_

        P=RL.learn()
        print(i_episode)
'''

if __name__ == '__main__':
    main()
    #test()
