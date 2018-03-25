#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import random
import numpy as np
from collections import deque
from keras.models import Sequential, clone_model
from keras.layers import Dense
from keras.optimizers import Adam

class QLearnerEvolverFlappy:

    def __init__(self, act_size ,state_size):

        self.state_size = state_size
        self.action_size = act_size

        self.memory = deque(maxlen=5000) #memory for replay
        self.gamma = 0.99    # discount rate
        self.epsilon = 0.1  # exploration rate
        self.should_epsilon_decay = True #False -> no epsilon decrease
                                         #True -> exponential decay
        self.epsilon_min = 0.0001
        self.epsilon_decay = 0.9999
        self.learning_rate = 0.0001
        self.batch_size = 32
        self.model = self._build_model()
        self.model_old = clone_model(self.model) #second model of DDQN
        self.model_old.set_weights(self.model.get_weights()) #same weights
        self.MAX_FEEDBACK = 10 #there are 30 frames between 2 pipes, so 10 is reasonable


    def _build_model(self):
        '''Neural Net for Deep-Q learning Model'''
        model = Sequential()
        model.add(Dense(100, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def pickAction(self,  state):
        '''epsilon greedy way to pick an action'''
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        predictions = self.model.predict(state)
        return np.argmax(predictions[0])  # returns action

    def updateEvolver(self):
        ''' update the fast model '''
        if len(self.memory) > self.batch_size:
            self.replay(self.batch_size)


    def updateModel(self):
        ''' update the slow model '''
        self.model_old.set_weights(self.model.get_weights()) #enables a deep copy

    def remember(self, state, action, reward, next_state, done):
        ''' remember the last steps, and take into account a feedback '''
        for i in range(1,min(len(self.memory), self.MAX_FEEDBACK)):
            s,a,r,ns,d = self.memory[-i]
            self.memory[-i] = (s,a,r+reward*self.gamma**i,ns,d)

        self.memory.append((state, action, reward, next_state, done))


    def replay(self, batch_size):
        ''' replay phase to train the fast model '''
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model_old.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        #espilon decaying
        if self.should_epsilon_decay:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
