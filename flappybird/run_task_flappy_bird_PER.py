#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Guide : Installation...
 http://pygame-learning-environment.readthedocs.io/en/latest/user/home.html
"""

from ple.games.flappybird import FlappyBird
from ple import PLE
from evolver_flappy_bird_PER import QLearnerEvolverFlappy, LearningPoint
import numpy as np
import time
import math
from collections import deque
from random import randint

#IF YOU PRESS U ONCE YOU CLICKED ON THE GAME SCREEN
#UI FREEZES BUT THE GAME RUNS MUCH FASTER
#THEN YOU JUST HAVE TO PRESS U AGAIN

def process_state(state):
    ''' process the variables that are returned by the environment '''
    state["next_next_pipe_bottom_y"] -= state["player_y"]
    state["next_next_pipe_top_y"] -= state["player_y"]
    state["next_pipe_top_y"] -= state["player_y"]
    state["next_pipe_bottom_y"] -= state["player_y"]
    state["previous_pipe_top_y"] -= state["player_y"]
    state["previous_pipe_bottom_y"] -= state["player_y"]
    return np.array([list(state.values())])


game = FlappyBird()
p = PLE(game, fps=30, display_screen=True, force_fps=False, state_preprocessor=process_state)
p.init()
game.ple = p
p.init()

#agent
action_set = p.getActionSet()
agent = QLearnerEvolverFlappy(len(action_set),p.getGameStateDims()[1])
agent.should_epsilon_decay = False

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
EXPLORE = 5000000 #small is 300000, big is 5000000
FINAL_EPSILON = 0.005
INITIAL_EPSILON = 0.01
INITIAL_BETA = 0.1
FINAL_BETA = 1

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

    observation = p.getGameState()
    action = agent.pickAction(observation)
    reward = p.act(p.getActionSet()[action])
    diff = abs(-agent.model.predict(observation)[0,1]+reward+agent.gamma * \
              np.amax(agent.model_old.predict(p.getGameState())[0]))
    agent.remember(observation, action, reward, p.getGameState(), p.game_over(), diff)

    #epsilon linear decrease
    if nb_frames<EXPLORE:
        agent.epsilon = (INITIAL_EPSILON*(EXPLORE-nb_frames)+FINAL_EPSILON*nb_frames)/EXPLORE
    else:
        agent.epsilon = FINAL_EPSILON

    if reward>0.5:
        score_game += 1

    #first we fill the memory of the agent and we don't train it
    if nb_frames==500:
        print("starting training")

    #for the fast model
    if nb_frames%10==0 and nb_frames>=500:
        agent.updateEvolver()
        last_losses.append(round(sum(abs((-agent.model.predict(observation)+reward+agent.gamma*agent.model.predict(p.getGameState()))[0])),3))

    #for the slow model
    if nb_frames%300==0 and nb_frames>=len(agent.memory):
        agent.updateModel()

    #to save the model
    if nb_frames%5000==0:
        print(agent.model.predict(observation))
        print("5000 frames, saving model")
        print("nb frames since beginning :", nb_frames)
        agent.save("weights_nn/flappy_new_features_100_"+str(number_experiment)+".h5")


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
        
    #print some info
    if nb_games%10==0 and not flag_game_10:
        print(nb_games)
        print("loss :", "inf" if len(last_losses)==0 else sum(last_losses)/len(last_losses))
        print("epsilon :", agent.epsilon)
        print("\n")
        flag_game_10 = True
    if nb_games%10==1:
        flag_game_10 = False
