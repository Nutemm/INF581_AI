#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Guide : Installation...
 http://pygame-learning-environment.readthedocs.io/en/latest/user/home.html
"""

from ple.games.flappybird import FlappyBird
from ple import PLE
from evolver_reinforce import PolicyNetwork
import numpy as np
import time
import math
from collections import deque
from random import randint

def process_state(state):
    ''' variables that are returned by the environment '''
    return np.array([list(state.values())])


game = FlappyBird()
p = PLE(game, fps=30, display_screen=True, force_fps=False, state_preprocessor=process_state)
p.init()
game.ple = p
p.init()


#agent
action_set = p.getActionSet()
agent = PolicyNetwork(len(action_set),p.getGameStateDims()[1])

#some flags and variables
nb_games = 1
nb_frames = 0
last_losses = deque(maxlen=1000)
flag_game_10 = False
flag_game_100 = False
flag_game_500 = False
score_game = 0
last_500_games_score = deque(maxlen=500)

#variables linked to epsilon decrease
EXPLORE = 300000 #small is 300000, big is 5000000
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.8

#a random number for score/model saving
number_experiment = randint(0,10000000)

while 1:

    nb_frames += 1

    if p.game_over():
        p.reset_game()
        nb_games += 1
        print("score for this game :", score_game)
        last_500_games_score.append(score_game)
        score_game = 0
        agent.reinforce()
        agent.forget_last_episode()

    if nb_frames<EXPLORE:
        agent.epsilon = (INITIAL_EPSILON*(EXPLORE-nb_frames)+FINAL_EPSILON*nb_frames)/EXPLORE
    else:
        agent.epsilon = FINAL_EPSILON

    observation = p.getGameState()
    action = agent.pickAction(observation)
    reward = p.act(p.getActionSet()[action])
    agent.store_transition(observation, action, reward)



    if reward>0.5:
        score_game += 1


    if nb_frames%5000==0:
        print("5000 frames, saving model")
        print("nb frames since beginning :", nb_frames)

    #for the curves
    if nb_games%500==0 and not flag_game_500:
        with open('results/file_score_nf_'+str(number_experiment)+'.txt', 'a') as the_file:
            the_file.write('\n'+str(sum(last_500_games_score)/500))
        flag_game_500 = True
    if nb_games%500==1:
        flag_game_500 = False


    if nb_games%100==0 and not flag_game_100:
        with open('results/file_loss_nf_'+str(number_experiment)+'.txt', 'a') as the_file:
            the_file.write('\n'+str(sum(np.array(last_losses)[-100:])/100))
        flag_game_100 = True
    if nb_games%100==1:
        flag_game_100 = False

    if nb_games%10==0 and not flag_game_10:
        print(nb_games)
        print("loss :", "inf" if len(last_losses)==0 else sum(last_losses)/len(last_losses))
        print("epsilon :", agent.epsilon)
        print("\n")
        flag_game_10 = True
    if nb_games%10==1:
        flag_game_10 = False
