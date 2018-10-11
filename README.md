# AI

This project contains different RL agents for various games that were implemented in Python through the pygame-learning-environment (https://pygame-learning-environment.readthedocs.io/en/latest/index.html).
The game we mainly worked on was FlappyBird.

The description and analyses of the agents we coded is available in report.pdf.

To run one of the agents, one can run one of the following scripts:

run_task_flappy_bird : for the normal DDQN agent (raises good scores)

run_task_flappy_bird_PER : for the DDQN agent with prioritized experience replay (raises good scores but trains slower because of the higher complexity of the training algorithm. However, this algorithm seems to be the most promising with a powerful enough computer.)

run_task_flappy_bird_random_search : for the agent with random search in the parameters (raises bad scores, hardly improves)

run_task_flappy_bird_reinforce : for the REINFORCE agent (raises bad scores, hardly improves)

Press 'u' (once you've clicked on the GUI) to make the GUI pauses and the training faster. Then 'u' again to resume GUI, and watch the improvements.

Sometimes the agent can fall in a really bad local optimum, which prevents it from training more and getting good scores. In this case, it is better to restart the whole training by re-running the script.


