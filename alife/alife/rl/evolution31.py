from alife.rl.agent import Agent
import numpy as np
import random
from collections import deque
from keras.models import Sequential, clone_model
from keras.layers import Dense
from keras.optimizers import Adam
from pathlib import Path

import copy


class Evolver31(Agent):
    '''
        A Policy-Search Method.

        All agents form this class will share a Q-network (Model global variable)
        The Q-network serves as an estimate of the action-value function Q
        This model is trained using the principle of experience replay
        and we keep a copy of the main network (Model2) that we update only
        every C steps (see our rapport for details)
    '''

    Model = None   #main Q-network
    Model2 = None   #keep a copy of that Q-network
    Update_every = 1000 #Model2 is updated every Update_every actions
    Memory = deque(maxlen=3000)   #will store sequences (s,a,r,s') for later experience replay

    Global_count = 0 #number of actions taken by agents of this class so far
    Exp_replay = 1000    #when Exp_replay actions have been taken, it launches an experience replay
    Num_replay = 300 #number of sequences replayed at each replay

    disc = 6  # discretisation parameter
    gamma = 0.98  # discount rate
    epsilon = 1.0  # exploration rate
    epsilon_min = 0.01  # minimal value for epsilon
    epsilon_decay = 0.99995
    learning_rate = 0.001  # learning rate in the network


    def __init__(self, obs_space, act_space, gen=1):
        """
            Init.


            Parameters
            ----------

            obs_space : BugSpace
                observation space
            act_space : BugSpace
                action space
            gen : int
                current generation

        """


        self.count = 1  #number of actions taken so far by this agent

        self.obs_space = obs_space
        self.act_space = act_space

        D = obs_space.shape[0]
        L = act_space.shape[0]
        self.state_size = D

        #We discretize the action space
        self.action_size = (Evolver31.disc)**L

        # This is just for visualization:
        self.generation = gen  # generation
        self.log = np.zeros((100000, D + L + 1))  # for storing
        self.t = 0  # for counting

        if(gen==1):

            Evolver31.Model = self._build_model()
            #If we previously saved our model in a file, we load it
            my_file = Path("network_31.h5")
            if my_file.is_file():
                self.load("network_31.h5")

            Evolver31.Model2 = clone_model(Evolver31.Model)
            Evolver31.Model2.set_weights(Evolver31.Model.get_weights()) #same weights

    def _build_model(self):
            # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu',name='first'))
        model.add(Dense(24, activation='relu',name='second'))
        model.add(Dense(self.action_size, activation='linear',name='third'))
        model.compile(loss='mse',optimizer=Adam(lr=Evolver31.learning_rate))
        return model

    def load(self, name):
        Evolver31.Model.load_weights(name)

    def save_model(self, name):
        Evolver31.Model.save_weights(name)
        print("Model 31 Saved !")


    def __str__(self):
        ''' Return a string representation (e.g., a label) for this agent '''
        return ("Evolver 31 (dql): Gen %d" % (self.generation))

    def act(self, obs, reward, done=False):
        """
            Act.

            Parameters
            ----------

            obs : numpy array of length D
                the state at the current time
            reward : float
                the reward signal at the current time

            Returns
            -------

            A number array of length L
                (the action to take)
        """

        #mettre en mémoire (s,a,r,s')
        if(hasattr(self,'previous_state')&hasattr(self,'previous_action')):
            self.remember(self.previous_state,self.previous_action,reward,obs)

        self.previous_state=obs


        # Save some info to a log
        Evolver31.Global_count+=1
        D = self.obs_space.shape[0]
        self.log[self.t, 0:D] = obs
        self.log[self.t, -1] = reward
        self.t = (self.t + 1) % len(self.log)

        # choix de l'action atemp dans l'espace discrétisé (epsilon-greedy)
        atemp = 0
        if np.random.rand() <= Evolver31.epsilon:
            atemp = random.randrange(self.action_size)
        else:
            act_values = Evolver31.Model.predict(np.array([obs]))
            atemp = np.argmax(act_values[0])

        #remember the action taken for later
        self.previous_action=atemp

        #convertir l'action de l'espace discrétisé à l'espace continu
        a = np.random.randn(2)
        a[0]=self.act_space.low[0] + (atemp//Evolver31.disc)* ((self.act_space.high[0]-self.act_space.low[0])/Evolver31.disc)
        a[1]=self.act_space.low[1] + (atemp%Evolver31.disc)*  ((self.act_space.high[1]-self.act_space.low[1])/Evolver31.disc)

        # More logging ...
        self.log[self.t, D:-1] = a

        #experience replay every Exp_replay actions
        if(Evolver31.Global_count%Evolver31.Exp_replay==0):
            print("training bug!")
            print("Epsilon value "+str(Evolver31.epsilon))
            self.replay(Evolver31.Num_replay)

        #Model update
        if(Evolver31.Global_count%Evolver31.Update_every==0):
            Evolver31.Model2.set_weights(Evolver31.Model.get_weights())  # same weights

        if(Evolver31.Global_count%1000==0):
            self.save_model("network_31.h5")

        # Return
        return a


    def remember(self, state, action, reward, next_state):
        Evolver31.Memory.append((state, action, reward, next_state))

    def replay(self, batch_size):
        minibatch = random.sample(Evolver31.Memory, batch_size)
        for state, action, reward, next_state in minibatch:
            target = (reward + Evolver31.gamma *
                              np.amax(
                                 Evolver31.Model2.predict(np.array([next_state]))[0]
                              )
                      )

            target_f = Evolver31.Model.predict(np.array([state]))
            target_f[0][action] = target
            Evolver31.Model.fit(np.array([state]), target_f, epochs=1, verbose=0)
            if Evolver31.epsilon > Evolver31.epsilon_min:
                Evolver31.epsilon *= Evolver31.epsilon_decay

    def spawn_copy(self):
        """
            Spawn.

            Returns
            -------

            A new copy (child) of this agent, [optionally] based on this one (the parent).
        """
        b = Evolver31(self.obs_space, self.act_space, self.generation + 1)
        b.generation = self.generation +1

        return b

    def save(self, bin_path, log_path, obj_ID):
        """
            Save a representation of this agent.

            Here we save a .csv of the state/action/reward signals.
            (such that it could be loaded later by pandas for visualization).
        """
        header = [("X%d" % j) for j in range(self.log.shape[1])]
        header[-1] = "reward"
        fname = log_path + ("/%d-%s-G%d.log" % (obj_ID, self.__class__.__name__, self.generation))
        np.savetxt(fname, self.log[0:self.t, :], fmt='%4.3f', delimiter=',', header=','.join(header), comments='')
        print("Saved log to '%s'." % fname)


