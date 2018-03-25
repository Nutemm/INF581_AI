#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import random
import numpy as np
from collections import deque
from keras.models import Sequential, clone_model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.initializers import RandomNormal

class RandomSearch:

    def __init__(self, act_size ,state_size):

        self.state_size = state_size
        self.action_size = act_size
        self.best_model = None
        self.best_score = 0
        self.model = self._build_model()
        self.epsilon = 0.01

        #for hill climbing:
        self.alpha = 0.3
        self.decay_alpha = 0.999

    def _build_model(self):
        '''Neural Net for Deep-Q learning Model'''
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.2, seed=None),
                bias_initializer='zeros'))
        model.add(Dense(self.action_size, activation='linear',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=0.2, seed=None),
                  bias_initializer='zeros'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=0.1))
        return model

    def pickAction(self,  state):
        '''epsilon greedy way to pick an action'''
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        predictions = self.model.predict(state)
        return np.argmax(predictions[0])

    def new_Evolver(self, is_hill_climbing = False):
        ''' create a new evolver, either coming from the previous one or
        completely independant '''
        if not is_hill_climbing: #Random search
            self.model = self._build_model()
        else: #Hill climbing
            for l in range(2):
                ar_weights = self.best_model.layers[l].get_weights()
                for i in range(len(ar_weights)):
                    for j in range(len(ar_weights[i])):
                        ar_weights[i][j] += np.random.normal(0, self.alpha, 1)
                self.model.layers[l].set_weights(ar_weights)
            self.alpha *= self.decay_alpha


    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
