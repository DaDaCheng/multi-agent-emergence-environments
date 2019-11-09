#!/usr/bin/env python3
import logging
#import click
import numpy as np
import argparse
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

# from simphas.MRL import mpolicy

# import gym
# from RL_brain_2 import PolicyGradient
from RL_brain_3 import PolicyGradientAgent

# import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--episode', type=int, default=350)
parser.add_argument('--n_episode', type=int, default=10000)
parser.add_argument('--opt', default='SGLD')
parser.add_argument('--n_hiders', type=int, default=1)
parser.add_argument('--n_seekers', type=int, default=1)
parser.add_argument('--n_agents', type=int, default=2)
parser.add_argument('--seeds', type=int, default=1)
parser.add_argument('--out', default='output')
parser.add_argument('--speed', type=int, default=1)
args = parser.parse_args()


def edge_punish(x, y, l=0.2, p=3.53, w=0):
    xx = 0.0
    if (np.abs(x - 0) < l) | (np.abs(x - p) < l):
        xx = xx + 1.0
    elif (np.abs(y - 0) < l) | (np.abs(y - p) < l):

        xx = xx + 1.0

    return w * xx * 1.0


def matdis(n, obs_x):
    dism = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dism[i, j] = np.sqrt(np.sum((obs_x[i, :2] - obs_x[j, :2]) ** 2))
    return dism


def matmas(n, mas):
    matm = np.empty([n, n], dtype=bool)
    for i in range(n):
        for j in range(n):
            if i > j:
                matm[i, j] = mas[i, j]
            elif i < j:
                matm[i, j] = mas[i, j - 1]
            else:
                matm[i, j] = False
    return matm


def game_rew(n,n_seekers, dism, matm, thr=1.0):
    return np.sum( (np.sum ( ((dism < np.ones((n,n))*thr) & (matm))[-n_seekers:], axis=0)>0))




env_name = 'mae_envs/envs/mybase.py'

display = True
n_agents= args.n_agents
n_seekers=args.n_seekers
n_hiders=args.n_hiders
episode=args.episode
n_episode=args.n_episode
kwargs = {}
seed=args.seeds
opt=args.opt
lr=args.learning_rate
out=args.out
speed=args.speed

kwargs.update({
    'n_agents': n_agents,
    'n_seekers': n_seekers,
    'n_hiders': n_hiders,
    'n_boxes':0,
    'cone_angle': 2 * np.pi,
    #'n_substeps' : 1

})




module = run_path(env_name)
make_env = module["make_env"]
args_to_pass, args_remaining = extract_matching_arguments(make_env, kwargs)
env = make_env(**args_to_pass)
env.reset()
#env_viewer = EnvViewer(env)


rhlist=[]
rslist=[]

def main(sk=None,hd=None,vlag=0):

    '''
    RL = mpolicy(
        n_actions=9,
        n_features=8,
        #n_features=4,
        learning_rate=0.01,
        reward_decay=0.9,
        units=30
        # output_graph=True,
    )
    '''
    '''
    Hider = PolicyGradient(
        n_actions=9,
        n_features=4,
        learning_rate=0.01,
        reward_decay=0.99,
        policy_name=Hpolicy_name
        # output_graph=True,
    )

    Seeker = PolicyGradient(
        n_actions=9,
        n_features=4,
        learning_rate=0.01,
        reward_decay=0.99,
        policy_name=Spolicy_name
        # output_graph=True,
    )
    '''
    if vlag == 0:
        Seeker = PolicyGradientAgent(lr, [8], n_actions=9, layer1_size=20, layer2_size=10,opt=opt,seed=seed)

        Hider = PolicyGradientAgent(lr, [8], n_actions=9, layer1_size=20, layer2_size=10,opt=opt,seed=seed+12345)
    else:
        Seeker = sk
        Hider = hd
    rs = []
    rh = []
    for ii in range(n_episode):
        # env_viewer.env_reset()
        env.reset()
        sampleaction = np.array([[5, 5, 5], [5, 5, 5]])
        action = {'action_movement': sampleaction}

        # obs, rew, down, _ = env_viewer.step(action)
        obs, rew, down, _ = env.step(action)
        observation = np.array(
            [obs['observation_self'][0][0], obs['observation_self'][0][1], obs['observation_self'][0][4],
             obs['observation_self'][0][5], obs['observation_self'][1][0], obs['observation_self'][1][1],
             obs['observation_self'][1][4], obs['observation_self'][1][5]])
        # observation = np.array([obs['observation_self'][1][0], obs['observation_self'][1][1],obs['observation_self'][1][4],obs['observation_self'][1][5]])

        for i in range(episode):
            action_Seeker = Seeker.choose_action(observation)
            action_Hider = Hider.choose_action(observation)
            # print(action_Seeker)

            if np.random.rand() > 0.95:
                action_Hider = np.random.randint(9)
            h1 = (action_Hider // 3 - 1) * 1 + 5
            h2 = (action_Hider % 3 - 1) * 1 + 5
            #h1,h2=5,5
            #print(action)

            if np.random.rand()>0.95:
                action_Seeker=np.random.randint(9)

            s1=(action_Seeker//3-1)*speed+5
            s2=(action_Seeker%3-1)*speed+5






            ac = {'action_movement': np.array([[h1, h2, 5], [s1, s2, 5]])}
            #print(ac)
            #obs_, reward, done, info = env_viewer.step(ac, show=False)
            obs_, reward, done, info = env.step(ac)
            observation_ = np.array([obs_['observation_self'][0][0], obs_['observation_self'][0][1],obs_['observation_self'][0][4], obs_['observation_self'][0][5],obs_['observation_self'][1][0], obs_['observation_self'][1][1], obs_['observation_self'][1][4], obs_['observation_self'][1][5]])


           # if not obs_['mask_aa_obs'][1][0]:
            #    rew= 5.0-( np.sqrt((observation_[4] - observation_[0]) ** 2 + (observation_[5] - observation_[1]) ** 2)*5)
            #else:
            #    rew= 5.0-( np.sqrt((observation_[4] - observation_[0]) ** 2 + (observation_[5] - observation_[1]) ** 2)*5)
            rew=1.0/np.sqrt((observation_[4] - observation_[0]) ** 2 + (observation_[5] - observation_[1]) ** 2)*5-3
            #print(observation_)

            #rrew=3-edge_punish(observation_[0],observation_[1])
            #print(observation_[0],observation_[1])
            #Seeker.store_transition(observation[-4:], action_Seeker, +rew)
            Seeker.store_rewards(rew-edge_punish(observation_[4],observation_[5]))
            Hider.store_rewards(-rew-edge_punish(observation_[0],observation_[1]))
            #print(-rrew)
            #Hider.store_transition(observation[4:], action_Hider, rrew)

            #print(50-edge_punish(observation_[0],observation_[1]))
            observation = observation_
        print(ii)
        #print(np.mean(Seeker.reward_memory[0]))
        rs.append(np.mean(Seeker.reward_memory))
        rh.append(np.mean(Hider.reward_memory))
        if vlag == 0:
            Hider.learn()
            Seeker.learn()
        else:
            Seeker.reward_memory = []
            Seeker.action_memory = []
            Hider.reward_memory = []
            Hider.action_memory = []


    #np.save('~/Downloads/'+out+'.npy', a)
    #rhlist.append(rh)
    #rslist.append(rs)
    np.save('output/'+opt+'_h_'+out + '.npy', rh)
    np.save('output/'+opt+'_s_'+out + '.npy', rs)
    # print(ii,R)
    return Seeker, Hider

if __name__ == '__main__':
    # S2, H2 = main(output='2', speed=2)
    # S3, H3 = main(output='3', speed=3)
    # S4, H4 = main(output='4', speed=4)
    # S1, H1 = main(output='1', speed=1)
    S1, H1 = main()
    # import pickle
    # pickle_file = open('objS2.pkl', 'wb')
    # pickle.dump(S2, pickle_file)
    # pickle_file.close()

    # pickle_file = open('objH2.pkl', 'wb')
    # pickle.dump(H2, pickle_file)
    # pickle_file.close()

    # main(sk=S1, hd=H4, output='41', speed=1, vlag=1)
    # main(sk=S1, hd=H3, output='31', speed=1, vlag=1)
    # main(sk=S1, hd=H2, output='21', speed=1, vlag=1)
    # test()

    #np.save('SGLDS.npy', rslist)
    #np.save('SGLDH.npy', rhlist)