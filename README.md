# Reinforcement Learning
- The project: reinforcement_learning
- Description: Generative Dungeon Maps
- Data Source: Imki
- Type of analysis: q-learning algorithm


- The steps to deploy and reproduce your results:
    The file can be found in : reinforcement_learning/RL_pygame_class.py

    I used vs code to write the code in python with a virtual ubuntu environment.

    Don't forget to pip install pygame.

    To read the file: python RL_pygame_class.py

    To have as an executable: pyinstaller RL_pygame_class.py --onefile --noconsole

- Your architecture’s presentation:
    I wrote a class modified from the [code] (https://becominghuman.ai/q-learning-a-maneuver-of-mazes-885137e957e4) and [github](https://github.com/bvpsk/Reinforcement-learning/blob/master/Q-Learning/git_simple_game.py).

    There is a starting point, a treasure point and an exit. Four walls are also added.

    The agent finds the shortest way passng by the treasure down to the exit.

    The visualisation is done with pygame.

- All the information that need to be mentioned:
    Each line of code is commented in the main document.

    The folder notebooks contains testing jupyter notebook files which helped me to learn.

- BONUS: discuss an evaluation metric to assess the difficulty of the maze (it doesn’t
need to be implemented).
    We can use evaluation metrics such as max, min, average, moving average. See how they are evolving
    with different epsilon or epochs.
