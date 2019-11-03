

import numpy as np
import tensorflow as tf


# reproducible
#np.random.seed(1)
#tf.set_random_seed(1)

class hiderpolicy(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001, units=20):
        self.n_features=n_features
        self.n_actions=n_actions
        self.sess = sess

        self.lr=lr
        self.units = units
        self._build_net()
    def _build_net(self):
        with tf.name_scope('hider_inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, None, name="actions_numm")
            self.tf_vt = tf.placeholder(tf.float32, None, name="actions_value")

            # fc2
        with tf.variable_scope('hider'):
            layer = tf.layers.dense(
                inputs=self.tf_obs,
                units=self.units,
                activation=tf.nn.tanh,  # tanh activation
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                bias_initializer=tf.constant_initializer(0.0),
                name='h1'
            )
            # fc2
            all_act = tf.layers.dense(
                inputs=layer,
                units=self.n_actions,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                bias_initializer=tf.constant_initializer(0.0),
                name='h2'
            )

        self.all_act_prob = tf.nn.softmax(all_act, name='hide_prob')

        with tf.name_scope('hider_loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
            # or in this way:
           #neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(-neg_log_prob * self.tf_vt)  # reward guided loss

        with tf.name_scope('hider_train'):
            #self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(loss)
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action, prob_weights


    def learn(self,obs,act,rew):
        # discount and normalize episode reward
        #discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        self.sess.run(self.train_op, feed_dict={
             self.tf_obs: np.vstack(obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(act),  # shape=[None, ]
             self.tf_vt: rew,  # shape=[None, ]
        })

        #self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return


class seekerpolicy(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001,units=20):
        self.n_features=n_features
        self.n_actions=n_actions
        self.sess = sess
        self.lr = lr
        self.units=units
        self._build_net()
    def _build_net(self):
        with tf.name_scope('seeker_inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, None, name="actions_numm")
            self.tf_vt = tf.placeholder(tf.float32, None, name="actions_value")

            # fc2
        with tf.variable_scope('seeker'):
            layer = tf.layers.dense(
                inputs=self.tf_obs,
                units=self.units,
                activation=tf.nn.tanh,  # tanh activation
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=1),
                bias_initializer=tf.constant_initializer(0.0),
                name='s1'
            )
            # fc2
            all_act1 = tf.layers.dense(
                inputs=layer,
                units=self.units,
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=1),
                bias_initializer=tf.constant_initializer(0.0),
                name='s1.5'
            )


            all_act = tf.layers.dense(
                inputs=all_act1,
                units=self.n_actions,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=1),
                bias_initializer=tf.constant_initializer(0.0),
                name='s2'
            )

        self.all_act_prob = tf.nn.softmax(all_act, name='seeker_prob')

        with tf.name_scope('seeker_loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
            # or in this way:
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

        with tf.name_scope('seeker_train'):
            #self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(loss)
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)


    def choose_action(self, observation):
        lamb=1.0
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        prob_weights=prob_weights*lamb+(1.-lamb)/self.n_actions
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action, prob_weights


    def learn(self,obs,act,rew):
        # discount and normalize episode reward
        #discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        self.sess.run(self.train_op, feed_dict={
             self.tf_obs: np.vstack(obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(act),  # shape=[None, ]
             self.tf_vt: rew,  # shape=[None, ]
        })

        #self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return
    def test(self,observation):
        print(self.sess.run(self.tf_obs,feed_dict={self.tf_obs: observation[np.newaxis, :]}))
        return
class mpolicy(object):
    def __init__(
            self,
            n_actions,
            n_features,
            n_agents=2,
            n_hiders=1,
            n_seekers=1,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
            units=10
    ):
        self.units=units
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.n_agents=n_agents
        self.n_hiders=n_hiders
        self.n_seekers = n_seekers


        self.ep_obs, self.ep_as_h, self.ep_as_s, self.ep_rs =[] , [], [],[]



        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)


        self._build()
        self.sess.run(tf.global_variables_initializer())

    def _build(self):
        self.hp=hiderpolicy(self.sess, self.n_features, self.n_actions, lr=self.lr, units=self.units)
        self.sp=seekerpolicy(self.sess, self.n_features, self.n_actions, lr=self.lr)


    def get_action(self, obs):
        hpaction,hpa=self.hp.choose_action(obs)
        spaction,spa=self.sp.choose_action(obs)
        #self.sp.test(obs)
        self.ep_obs.append(obs)
        self.ep_as_h.append(hpaction)
        self.ep_as_s.append(spaction)
        return hpaction, spaction

    def get_rew(self, rew):
        self.ep_rs.append(rew)

    def action(self,obs,speed=1):
        h,s=self.get_action(obs)

        '''
        678
        345
        012
        '''
        def tra(data,dim=3):
            return [data//dim,data%dim,1]
        #return {action_movement: [tra(h), tra(s)]}
        return (np.array([tra(h), tra(s)])-1)*speed+5
    def learn(self):
        T=len(self.ep_rs)
        assert T == len(self.ep_obs) & T == len(self.ep_rs), 'shabileba'

        G=self._discount_and_norm_rewards(self.ep_rs)



        self.sp.learn(self.ep_obs,  self.ep_as_s,  G)
        GR=np.mean(self.ep_rs)

        self.ep_obs, self.ep_as_s, self.ep_as_h, self.ep_rs = [], [], [], []


        '''

        Glist=[]
        for t in range(T):

            G = 0.0
            for k in range(t + 1, T + 1):
                G += (self.gamma ** (k - t - 1)) * self.ep_rs[k - 1]
            G = G * (self.gamma ** t)
            Glist.append(G)

        self.sp.learn(self.ep_obs, self.ep_as_s, Glist)

        GR=np.sum(self.ep_rs)
        self.ep_obs, self.ep_as_s, self.ep_as_h, self.ep_rs = [], [], [],[]

        '''


        return  GR

    def _discount_and_norm_rewards(self,rlsit):
        # discount episode rewards
        rlsit=np.array(rlsit,dtype='float64')
        discounted_ep_rs = np.zeros_like(rlsit)
        running_add = 0.0
        for t in reversed(range(0, len(rlsit))):
            running_add = running_add * self.gamma + rlsit[t]
            discounted_ep_rs[t] = running_add

        #for t in range(0, len(rlsit)):
        #    discounted_ep_rs[t] = discounted_ep_rs[t]*(self.gamma**t)

        # normalize episo   de rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs