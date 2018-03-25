#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Guide : Installation...
 http://pygame-learning-environment.readthedocs.io/en/latest/user/home.html
"""

from ple.games.flappybird import FlappyBird
from ple import PLE
from evolver_random_search import RandomSearch
import numpy as np
import time
import math
from collections import deque
from random import randint

def process_state(state):
    #print(state)
    return np.array([list(state.values())])


game = FlappyBird()
p = PLE(game, fps=30, display_screen=True, force_fps=False, state_preprocessor=process_state)
p.init()
game.ple = p
p.init()

#print(p.getActionSet())


#agent
action_set = p.getActionSet()
agent = RandomSearch(len(action_set),p.getGameStateDims()[1])

# agent.load("flappy1_100.h5")

nb_games = 1
nb_frames = 0
flag_game_10 = False
flag_game_100 = False
flag_game_50 = False
score_game = 0

last_50_games_score = deque(maxlen=50)

EXPLORE = 5000000 #small is 300000, big is 5000000
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.1

number_experiment = randint(0,10000000)

while 1:

    nb_frames += 1

    if p.game_over():
        p.reset_game()
        nb_games += 1
        print("score for this game :", score_game)
        last_50_games_score.append(score_game)
        score_game = 0

    observation = p.getGameState()
    #print(observation)
    #action = p.getActionSet()[randint(0,1)]
    action = agent.pickAction(observation)
    #print("action :", action)
    reward = p.act(p.getActionSet()[action])

    if nb_frames<EXPLORE:
        agent.epsilon = (INITIAL_EPSILON*(EXPLORE-nb_frames)+FINAL_EPSILON*nb_frames)/EXPLORE
    else:
        agent.epsilon = FINAL_EPSILON

    if reward>0.5:
        score_game += 1


    if nb_frames%5000==0:
        print("5000 frames, saving model")
        print("nb frames since beginning :", nb_frames)
        agent.save("weights_nn/flappy_new_features_100_"+str(number_experiment)+".h5")

    if nb_games%50==0 and not flag_game_50:
        if sum(last_50_games_score)/50>=agent.best_score:
            agent.best_score = sum(last_50_games_score)/50
            agent.best_model = agent.model
        agent.new_Evolver(is_hill_climbing=True)
        flag_game_50 = True
        print("new evolver")
        print("best evolver score since then :", agent.best_score)
        print("\n\n")

    if nb_games%50==1:
        flag_game_50 = False


    #for the curves
    # if nb_games%500==0 and not flag_game_500:
    #     with open('results/file_score_nf_'+str(number_experiment)+'.txt', 'a') as the_file:
    #         the_file.write('\n'+str(sum(last_500_games_score)/500))
    #     flag_game_500 = True
    # if nb_games%500==1:
    #     flag_game_500 = False
    #
    #
    # if nb_games%100==0 and not flag_game_100:
    #     with open('results/file_loss_nf_'+str(number_experiment)+'.txt', 'a') as the_file:
    #         the_file.write('\n'+str(sum(np.array(last_losses)[-100:])/100))
    #     flag_game_100 = True
    # if nb_games%100==1:
    #     flag_game_100 = False
