#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from heapq import heappush, heappop, heapify
import random
import numpy as np
from collections import deque
from keras.models import Sequential, clone_model
from keras.layers import Dense
from keras.optimizers import Adam

class LearningPoint:
    ''' class containing some infos about one decision of the agent,
    that are used in the training phase '''

    def __init__(self, state, action, reward, next_state, done, diff):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.diff = diff

    def __lt__(self, other):
        return self.diff > other.diff

    def return_var(self):
        return self.state, self.action, self.reward, self.next_state, self.done, self.diff


def heapify_point(l, index):
    '''
    In_place transformation for l (a heap) to put the element currently
    at the index position in l in its right place

    Returns a dict with the indices that have been swapped.
    keys -> old index
    value -> new index
    '''

    dict_changes = {}
    initial_index = index
    while True:
        if index==0:
            break
        parent_index = index//2-1 if index%2==0 else index//2
        if l[index]<l[parent_index]:
            dict_changes[parent_index]=index
            l[index], l[parent_index] = l[parent_index], l[index]
            index=parent_index
        else:
            break
    dict_changes[initial_index] = index
    return dict_changes



class QLearnerEvolverFlappy:

    def __init__(self, act_size ,state_size):

        self.state_size = state_size
        self.action_size = act_size
        self.memory = [] #memory for replay, we use it as a heap
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

        #indices of the last points in the heap (self.memory)
        self.last_points_index = deque(maxlen=self.MAX_FEEDBACK) #index of the

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

    def remember(self, state, action, reward, next_state, done, diff):
        ''' remember the last steps, and take into account a feedback '''

        #feedback for the last points
        for i in range(1, len(self.last_points_index)+1):
            j = self.last_points_index[-i]
            if j!=5000: #was deleted
                s,a,r,ns,d,diff = self.memory[j].return_var()
                self.memory[j] = LearningPoint(s,a,r+reward*self.gamma**i,ns,d,diff)

        #adding the new point
        self.memory.append(LearningPoint(state, action, reward, next_state, done, diff))
        self.last_points_index.append(len(self.memory)-1)
        changes = heapify_point(self.memory, len(self.memory)-1)
        self.take_into_account(changes, self.last_points_index)

        #limiting the size
        if len(self.memory)>5000:
            del(self.memory[-1])

    def replay(self, batch_size):
        ''' replay important events, using the rank of elements in the heap
         which represents a more or less sorted list,
         and train the fast model '''

        #compute minibatch with taking into account ranks of elements
        denom = sum(np.array([1/(i+10) for i in range(len(self.memory))]))
        probs = np.array([1/(i+10)/denom for i in range(len(self.memory))])
        minibatch_index = np.random.choice([i for i in range(len(self.memory))], size = batch_size, p=probs)
        minibatch = [self.memory[i] for i in minibatch_index]

        #training for the points in minibatch
        for i, point in enumerate(minibatch):
            reward, state, action, done, next_state = point.reward, point.state, \
                                    point.action, point.done, point.next_state
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model_old.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)

            #we change their position in the heap as the agent has been trained
            new_diff = abs(-self.model.predict(state)[0,action]+reward+self.gamma * \
                      np.amax(self.model_old.predict(next_state)[0]))
            self.memory[minibatch_index[i]].diff = new_diff
            changes = heapify_point(self.memory, minibatch_index[i])
            self.take_into_account(changes, self.last_points_index)

        #epsilon decaying
        if self.should_epsilon_decay:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay


    def take_into_account(self, changes, l):
        ''' changes is a dict with indices to change
        l may contain some of those indices, so it changes the ones that are in l
         '''
        for key, value in changes.items():
            for j in range(len(l)):
                if l[j]==key:
                    l[j]=value

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
