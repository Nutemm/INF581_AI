import numpy as np
import tensorflow as tf
import gym
from collections import deque
import matplotlib.pyplot as plt
import random


# reproducible
np.random.seed(1)
tf.set_random_seed(1)

env = gym.make('CartPole-v0')
env.seed(1)

RENDER = False


class PolicyNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
    ):
        self.n_actions = n_actions
        self.n_features = n_features

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self.picked_actions_prob = []

        self.gamma = 0.99

        self.alpha = 0.02

        self.alpha_decay = 0.99

        self.cross_entropy = None

        self.epsilon = 0.01

        self._build_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):

        # Placeholder for inputs (states)
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="number_of_the_actions")
            self.tf_rt = tf.placeholder(tf.float32, [None, ], name="Rt_values")

        # Hidden layer (l1)
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=10,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # Linear Layer (l2)
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        # softmax converts to probability
        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')

        #here we define the loss and the minimizer

        #cross_entropy matches the taken action with prob(action|state) so that it returns log(prob(taken action|state))
        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)
        #sum over all t
        self.loss = tf.reduce_sum(self.cross_entropy * self.tf_rt)
        #we build the minimizer
        self.minimizer = tf.train.AdamOptimizer(self.alpha).minimize(self.loss)

    # Chose an action according to network probability
    def pickAction(self, observation):
        if np.random.rand() <= self.epsilon:
            return 1
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s[0])
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def compute_loss(self):
        Rt = np.array([self.compute_Rt(t) for t in range(len(self.ep_obs))])
        return self.sess.run(self.minimizer, feed_dict={
             self.tf_obs: np.array(self.ep_obs),
             self.tf_acts: np.array(self.ep_as),
             self.tf_rt: Rt,
        })


    def compute_Rt(self, t):
        Rt = 0
        t_prime = t
        while len(self.ep_obs)>t_prime:
            Rt += self.gamma**(t_prime-t)*self.ep_rs[t_prime]
            t_prime += 1
        return Rt

    def forget_last_episode(self):
        self.ep_as, self.ep_obs, self.ep_rs, self.picked_actions_prob = [], [], [], [] #make the agent forget last game

    def reinforce(self):

        Rt = np.array([self.compute_Rt(t) for t in range(len(self.ep_obs))])

        #run the minimizer
        self.sess.run(self.minimizer, feed_dict={
             self.tf_obs: np.array(self.ep_obs),
             self.tf_acts: np.array(self.ep_as),
             self.tf_rt: Rt,
        })


    def decay(self):
        self.alpha *= self.alpha_decay
