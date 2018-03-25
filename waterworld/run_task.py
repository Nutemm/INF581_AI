"""
Guide : Installation...
 http://pygame-learning-environment.readthedocs.io/en/latest/user/home.html
 
 Please press u in the window if you wish to speed-up or slow-down the simulation
"""

from waterworld import WaterWorld
from ple import PLE
from evolver import QLearnerEvolver
import numpy as np

# Start the environment
NUM_CREEPS = 30
game = WaterWorld(width=500, height=500, num_creeps=NUM_CREEPS)

p = PLE(game, fps=30, display_screen=True, add_noop_action=False, force_fps=False)
game.ple = p
p.init()

# Start the agent
action_set = p.getActionSet()
agent = QLearnerEvolver(len(action_set))
p.state_preprocessor = agent.process_state

#agent.load("model.h5") 
#agent.epsilon = 0.05

fail,catch,j = 0,0,0
best_score = -np.inf
nb_games = 1

while 1:
    j+=1
    
    # On réinitialise de temps en temps
    if p.game_over() or j==50000:
        fail,catch,j=0,0,0
        best_score = max(best_score,p.score())
        nb_games+=1
        p.reset_game()

    observation = p.getGameState()
    action = agent.pickAction(observation)
    reward = p.act(action_set[action])
    
    if reward < -0.5:
        fail+=1
    if reward > 0.5:
        catch+=1
        
    agent.remember(observation, action, reward, p.getGameState(), p.game_over())
    
    if j%300==0:
        print("Game", nb_games,"/ Score :",p.score(),"/ Best :",best_score,"/ Success :",catch,"/ Fail :",fail,"/ Epsilon :",np.round(agent.epsilon,2))
        agent.updateEvolver()

    if j%3000==0:
        agent.updateModel()

    if j%10000==0:
        agent.save("model.h5")
