#!/usr/bin/env python3

import numpy as np
import argparse

from RL_brain_3 import PolicyGradientAgent
from lineenv import Lineenv


parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=1e-2, help='learning rate')
parser.add_argument('--GAMMA', type=float, default=0.99)
parser.add_argument('--episode', type=int, default=50)
parser.add_argument('--n_episode', type=int, default=1000)
parser.add_argument('--opt', default='SGLD')
parser.add_argument('--n_hiders', type=int, default=1)
parser.add_argument('--n_seekers', type=int, default=1)
parser.add_argument('--n_agents', type=int, default=2)
parser.add_argument('--seeds', type=int, default=123432)
parser.add_argument('--out', default='output')
parser.add_argument('--s_speed', type=int, default=1)
parser.add_argument('--h_speed', type=int, default=1)
parser.add_argument('--fileseeker', default='policy.pkl')
parser.add_argument('--filehider', default='policy.pkl')
parser.add_argument('--vlag',type=int, default=0)
args = parser.parse_args()


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
s_speed=args.s_speed
h_speed=args.h_speed
sfile=args.fileseeker
hfile=args.filehider
vlag=args.vlag
GAMMA=args.GAMMA





env = Lineenv(100)
env.reset()



def main(sk=None,hd=None,vlag=0):
    rhlist = []
    rslist = []

    Seeker=sk
    Hider=hd

    if Seeker == None:
        Seeker = PolicyGradientAgent(lr, [2], n_actions=2, layer1_size=2, layer2_size=2,opt=opt,seed=seed,GAMMA=GAMMA)
    if Hider == None:
        Hider = PolicyGradientAgent(lr, [2], n_actions=2, layer1_size=2, layer2_size=2,opt=opt,seed=seed+12345, GAMMA=GAMMA)

    for ii in range(n_episode):

        env.reset()


        sampleaction = np.array([1,1])

        observation= env.step(sampleaction)

        for i in range(episode):
            action_Seeker = Seeker.choose_action(observation)
            action_Hider = Hider.choose_action(observation)

            ac=[action_Hider,action_Seeker]
            #print(ac)
            ac = [2, action_Seeker]





            observation_  = env.step(ac)

            rew=abs(observation_[0]-observation_[1])*1.0

            Seeker.store_rewards(-rew)
            Hider.store_rewards(rew)

            observation = observation_
            env.render()
        print(ii)

        Gh=0
        Gs=0
        for i in reversed(range(episode)):
            Gh=Gh*GAMMA+Hider.reward_memory[i]
            Gs = Gs * GAMMA + Seeker.reward_memory[i]
        print(Gs)


        rhlist.append(Gh)
        rslist.append(Gs)
        if vlag == 0:
            Hider.learn()
            Seeker.learn()
        else:
            Seeker.reward_memory = []
            Seeker.action_memory = []
            Hider.reward_memory = []
            Hider.action_memory = []

    np.save('output/'+opt+'_h_'+out + '.npy', rhlist)
    np.save('output/'+opt+'_s_'+out + '.npy', rslist)
    return Seeker, Hider

if __name__ == '__main__':
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
        pickle_file = open('policys_'+out+'.pkl', 'wb')
        pickle.dump(S1, pickle_file)
        pickle_file.close()

        pickle_file = open('policyh_' + out + '.pkl', 'wb')
        pickle.dump(H1, pickle_file)
        pickle_file.close()

