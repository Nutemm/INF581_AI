#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import random
import numpy as np
from collections import deque
from keras.models import Sequential, clone_model
from keras.layers import Dense
from keras.optimizers import Adam
import keras
from utils import angle


class QLearnerEvolver:

    def __init__(self, act_size):

        self.action_size = act_size
        self.memory = deque(maxlen=300)
        
        # Stae preprocessor constants
        self.EYES_RANGE = 100
        self.N_EYES = 20
        self.state_size = 2+4*self.N_EYES

        
        # Q Learner Constant
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.4
        self.epsilon_decay = 0.999
        self.epsilon_seuil = 0.11
        self.learning_rate = 0.001
        
        # NN Constant
        self.batch_size = 32
        self.model = self._build_model()
        self.model_old = clone_model(self.model) #same shape of model
        self.model_old.set_weights(self.model.get_weights()) #same weights
        
        # Remerber Constant for a better Q-Learner
        self.MAX_FEEDBACK = 4
        self.gamma_feedback = 0.8

    def _build_model(self):
        # Build our NN Model
        model = Sequential()
        model.add(Dense(100, input_shape=(self.state_size,),activation='relu'))
       # model.add(Dense(50, activation='relu'))
        model.add(Dense(self.action_size, activation='linear',kernel_regularizer=keras.regularizers.l2(0.01)))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        model.summary()
        return model
    
    def process_state(self,state):
    
        positions = [state['player_x'], state['player_y'], state['player_velocity_x'], state['player_velocity_y']]
    
        positions_good = state['creep_pos']['GOOD'] # Positions
        positions_bad = state['creep_pos']['BAD']
        dist_good = state['creep_dist']['GOOD'] # Distances to player
        dist_bad = state['creep_dist']['BAD']
        vel_good = state['creep_vel']['GOOD'] # Velocities
        vel_bad = state['creep_vel']['BAD']
    
        creeps = []
        
        # Creation of useful features from creeps
        for creep,dist,v in zip(positions_good, dist_good,vel_good):
            if(dist<self.EYES_RANGE):
                pos_relative_creep = np.array([creep[0]-positions[0], creep[1]-positions[1]])
                vel = np.array([v[0]-positions[2],v[1]-positions[3]])
                vel_relative = 1/dist*np.dot(pos_relative_creep,vel)
                creeps.append(("GOOD",dist, int(self.N_EYES*angle(pos_relative_creep)/360),vel_relative))
        for creep,dist,v in zip(positions_bad, dist_bad, vel_bad):
            if(dist<self.EYES_RANGE):
                pos_relative_creep = np.array([creep[0]-positions[0], creep[1]-positions[1]])
                vel = np.array([v[0]-positions[2],v[1]-positions[3]])
                vel_relative = 1/dist*np.dot(pos_relative_creep,vel)
                creeps.append(("GOOD",dist, int(self.N_EYES*angle(pos_relative_creep)/360),vel_relative))
        creeps = sorted(creeps,key=lambda x : x[1])
        
        # An EYE is an angle range of 2*PI/N_EYES, each eye see the closest creep in its range if it's not too far
        eyes_vision = np.zeros((self.N_EYES, 4)) # Distance, is_good, is_bad, Velocity
        eyes_vision[:,0] = np.array([0]*self.N_EYES)
    
        for creep in creeps:
            if eyes_vision[creep[2],1]==0 and eyes_vision[creep[2],2]==0:
                eyes_vision[creep[2],0]=creep[1]
                eyes_vision[creep[2],1]=int(creep[0]=="GOOD")
                eyes_vision[creep[2],2]=int(creep[0]=="BAD")
                eyes_vision[creep[2],3]=creep[3]
    
        return np.array([positions[2:]+list(eyes_vision.flatten())])


    def pickAction(self,  state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        predictions = self.model.predict(state)
        return np.argmax(predictions[0])  # returns action

    def updateEvolver(self):
        # Frequently called by run_task to update the Fast NN
        if len(self.memory) > self.batch_size:
            self.replay(self.batch_size)

    def updateModel(self):
        # Called by run_task to update the slow NN
        self.model_old.set_weights(self.model.get_weights()) #enables a deep copy

    def remember(self, state, action, reward, next_state, done):
        for i in range(1,min(len(self.memory), self.MAX_FEEDBACK)):
            s,a,r,ns,d = self.memory[-i]
            self.memory[-i] = (s,a,r+reward*self.gamma_feedback**i,ns,d)

        self.memory.append((state, action, reward, next_state, done))
            

    def replay(self, batch_size):
        # take a sample of the last data and try to improve NN using this data
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                predictions = self.model_old.predict(next_state)[0]
                target = reward+ self.gamma * np.amax(predictions)
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
            
        # We set epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        if self.epsilon < self.epsilon_seuil:
            self.epsilon_decay = 0.99

    def load(self, name):
        # Load a pre-trained NN
        self.model.load_weights(name)

    def save(self, name):
        # Save the trained NN for later used
        self.model.save_weights(name)
