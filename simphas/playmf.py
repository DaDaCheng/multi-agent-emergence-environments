#!/usr/bin/env python3
import logging
#import click
import numpy as np
import argparse
import pickle
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
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--GAMMA', type=float, default=0.999,)
parser.add_argument('--episode', type=int, default=350)
parser.add_argument('--n_episode', type=int, default=100000)
parser.add_argument('--opt', default='SGLD')
parser.add_argument('--n_hiders', type=int, default=1)
parser.add_argument('--n_seekers', type=int, default=1)
parser.add_argument('--n_agents', type=int, default=2)
parser.add_argument('--seeds', type=int, default=1)
parser.add_argument('--out', default='output')
parser.add_argument('--s_speed', type=int, default=1)
parser.add_argument('--h_speed', type=int, default=1)
parser.add_argument('--fileseeker', default='policy.pkl')
parser.add_argument('--filehider', default='policy.pkl')
parser.add_argument('--outflag', default=0)
parser.add_argument('--vlag',type=int, default=0)
args = parser.parse_args()


def edge_punish(x, y, l=0.2, p=3.53, w=0):
    xx = 0.0
    if (np.abs(x - 0) < l) | (np.abs(x - p) < l):
        xx = xx + 1.0
    elif (np.abs(y - 0) < l) | (np.abs(y - p) < l):

        xx = xx + 1.0

    return w * xx * 1.0

'''multiple setting
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

'''


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
outflag=args.outflag
s_speed=args.s_speed
h_speed=args.h_speed
sfile=args.fileseeker
hfile=args.filehider
vlag=args.vlag
GAMMA=args.GAMMA

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
if display:
    env = EnvViewer(env)
    env.env_reset()



def main(sk=None,hd=None,vlag=0):
    rhlist = []
    rslist = []

    Seeker=sk
    Hider=hd

    if Seeker == None:
        Seeker = PolicyGradientAgent(lr, [8], n_actions=9, layer1_size=64, layer2_size=64,opt=opt,seed=seed,GAMMA=GAMMA)
    if Hider == None:
        Hider = PolicyGradientAgent(lr, [8], n_actions=9, layer1_size=64, layer2_size=64,opt=opt,seed=seed+12345, GAMMA=GAMMA)

    for ii in range(n_episode):
        if display:
            env.env_reset()
        else:
            env.reset()
        sampleaction = np.array([[5, 5, 5], [5, 5, 5]])
        action = {'action_movement': sampleaction}

        obs, rew, down, _ = env.step(action)
        observation = np.array(
            [obs['observation_self'][0][0], obs['observation_self'][0][1], obs['observation_self'][0][4],
             obs['observation_self'][0][5], obs['observation_self'][1][0], obs['observation_self'][1][1],
             obs['observation_self'][1][4], obs['observation_self'][1][5]])

        for i in range(episode):
            action_Seeker = Seeker.choose_action(observation)
            action_Hider = Hider.choose_action(observation)

            if np.random.rand() > 1:
                action_Hider = np.random.randint(9)
            h1 = (action_Hider // 3 - 1) * h_speed + 5
            h2 = (action_Hider % 3 - 1) * h_speed + 5

            if np.random.rand() > 1:
                action_Seeker=np.random.randint(9)
            s1=(action_Seeker//3-1)*s_speed+5
            s2=(action_Seeker%3-1)*s_speed+5






            ac = {'action_movement': np.array([[h1, h2, 5], [s1, s2, 5]])}
            #ac = {'action_movement': np.array([[5, 5, 5], [s1, s2, 5]])}

            obs_, reward, done, info = env.step(ac)
            observation_ = np.array([obs_['observation_self'][0][0], obs_['observation_self'][0][1],obs_['observation_self'][0][4], obs_['observation_self'][0][5],obs_['observation_self'][1][0], obs_['observation_self'][1][1], obs_['observation_self'][1][4], obs_['observation_self'][1][5]])

            rew1=np.sqrt((observation_[4] - observation_[0]) ** 2 + (observation_[5] - observation_[1]) ** 2)
            rew2=np.sqrt((observation_[0] -1.5) ** 2 + (observation_[1] - 1.5) ** 2)*5

            #print(rew)
            Seeker.store_rewards(-rew)
            Hider.store_rewards(-rew2+rew1)

            observation = observation_
        #print(ii)
        print('\r{}'.format(ii), end='')
        if outflag > 0:
            Gh=0
            Gs=0
            for i in reversed(range(episode)):
                Gh=Gh*GAMMA+Hider.reward_memory[i]
                Gs = Gs * GAMMA + Seeker.reward_memory[i]


        #rhlist.append(Gh)
        #rslist.append(Gs)

        rhlist.append(Hider.reward_memory)
        rslist.append(Seeker.reward_memory)

        if vlag == 0:
            Hider.learn()
            Seeker.learn()
        else:
            if outflag > 0:
                Seeker.reward_memory = []
                Seeker.action_memory = []
                Hider.reward_memory = []
                Hider.action_memory = []


    #np.save('~/Downloads/'+out+'.npy', a)
    #rhlist.append(rh)
    #rslist.append(rs)
    if outflag > 0:
        np.save('output/'+'h_'+out + '.npy', rhlist)
        np.save('output/'+'s_'+out + '.npy', rslist)
    # print(ii,R)
    return Seeker, Hider

if __name__ == '__main__':
    # S2, H2 = main(output='2', speed=2)
    # S3, H3 = main(output='3', speed=3)
    # S4, H4 = main(output='4', speed=4)
    # S1, H1 = main(output='1', speed=1)

    if vlag==0:
        sk=None
        hd=None
    elif vlag==1:
        sk=None
        hd=pickle.load(hfile)
    elif vlag==2:
        sk = pickle.load(hfile)
        hd = None
    else:

        with open(hfile, 'rb') as f:
            hd=pickle.loads(f.read())


        with open(sfile, 'rb') as f:
            sk=pickle.loads(f.read())



    S1, H1 = main(sk,hd)
    if vlag==0:
        pickle_file = open('policy_s_'+out+'.pkl', 'wb')
        pickle.dump(S1, pickle_file)
        pickle_file.close()

        pickle_file = open('policy_h_' + out + '.pkl', 'wb')
        pickle.dump(H1, pickle_file)
        pickle_file.close()

