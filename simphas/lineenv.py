import numpy as np
import time


#action={h,s}

class Lineenv(object):
    def __init__(self, linelen=20):
        self.linelen = linelen

        np.random.seed(100)
        self.hp = np.random.randint(linelen)
        self.sp = np.random.randint(linelen)


    def reset(self):
        self.hp = np.random.randint(self.linelen)
        self.sp = np.random.randint(self.linelen)

        return [self.hp, self.sp]

    def step(self,action):
        saction=action[1]
        if saction==0:
            if self.sp>0:
                self.sp=self.sp-1
            else:
                self.sp=0
        elif saction==1:

            if self.sp < (self.linelen-1):

                self.sp=self.sp+1

            else:
                self.sp=(self.linelen-1)



        haction=action[0]
        if haction==0:
            if self.hp>0:
                self.hp=self.hp-1
            else:
                self.sp=0
        elif haction==1:

            if self.hp < (self.linelen-1):

                self.hp=self.hp+1

            else:
                self.hp=(self.linelen-1)

        return [self.hp, self.sp]

    def render(self,FRESH_TIME=0):
        env_list = ['-'] * (self.linelen)
        if self.hp==self.sp:
            env_list[self.hp]='x'
        else:
            env_list[self.hp] = 'h'
            env_list[self.sp] = 's'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)