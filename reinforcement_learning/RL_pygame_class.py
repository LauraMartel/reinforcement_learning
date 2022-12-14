######## ----- Import librairies ------ ########
#pip install pygame
import numpy as np
import pygame
from time import time,sleep
from random import randint as r
import random
import pickle
import matplotlib.pyplot as plt


class dungeon_map:

    # Initializing function with the constants which will be used through the class
    def __init__(self, size, walls, learning_rate, discount, epsilon, reward_treasure, punition_walls, reward_exit):

        ######## --------Screen where the dungeon map will be displayed------- ########

        # choose the size of the maze. Here 4x4.
        self.n = size

        # number of walls you need
        self.penalities = walls

        # punition on the walls
        self.punition_walls = punition_walls

        # reward to pass by the treasure
        self.reward_treasure = reward_treasure

        # reward to reach the exit
        self.reward_exit = reward_exit

        # dispay a screen with pygame with size 4x4
        self.screen = pygame.display.set_mode((self.n*100,self.n*100))

        # background color. black here.
        self.background = (51,51,51)

        ######## --------Parameters for the reinforcement learning-------- ########
        # The learning rate or step size determines to what extent newly acquired information overrides old information.
        # ranges between 0 and 1
        self.alpha = learning_rate

        # weight measure of how important do we find future actions vs current action or future reward over current reward. values between 0 and 1
        self.gamma = discount

        # values between 0 and 1. The higher the epsilon, the more likely we are to perform a random action
        self.epsilon = epsilon

        ####### Movement
        self.actions = {"up": 0,"down" : 1,"left" : 2,"right" : 3}

        # initial position
        self.current_pos = [0,0]

        # initial q-learning table (Q) 16x4. 16 as 4 actions up, down, left, right
        self.Q = np.zeros((self.n**2,4))

        # colors for the background
        self.colors =  [(51,51,51) for i in range(self.n**2)]

        # empty list to be able to do the reinforcement learning
        self.terminals = []

        # initial rewards matrix with only -1 size 4x4 allows to penalize the path
        # and force the agent to go to the treasure
        # This can eventually be put as a user input.
        self.reward = -np.ones((self.n,self.n))

        # empty dictionnary used later to put the state
        self.current_state = {}


    # Function to create the pygame layout
    def layout(self):
        # exit, I put the reward for the exit to be the highest one. To force to go outside of treasure.
        self.reward[self.n-1,self.n-1] = self.reward_exit
        # treasure, I put -1 for the treasure for the ball to pass by the value, but not stay there.
        self.reward[2,1] = self.reward_treasure
        # color for the rectangle start
        self.colors[0] = (0,255,0)
        # color for the rectangle exit
        self.colors[self.n**2 - 1] = (0,255,0)
        # color for the rectangle treasure
        self.colors[self.n**2 - 7] = (253,255,71)


        # apply the penalties we have 4 walls
        while self.penalities != 0:
            i = r(0,self.n-1)
            j = r(0,self.n-1)
            # if the case in the reward array are equal to 0, we replace 4 walls by punition_walls
            # and put penalties randomly in the maze
            if self.reward[i,j] == -1 and [i,j] != [0,0] and [i,j] != [self.n-1,self.n-1] and [i,j]!=[1,2]:
                # punition for the walls which have the highest negative value
                self.reward[i,j] = self.punition_walls
                # keep at -1 for the 4 walls
                self.penalities -= 1
                # colors of the cases with penalties
                self.colors[self.n*i+j] = (255,0,0)  #red
                self.terminals.append(self.n*i+j)

        # draw grid in the pygame window
        c = 0
        for i in range(0,self.n*100,100):
            for j in range(0,self.n*100,100):
                # white rectangles with a size bigger than the black rectangles to draw the borders
                pygame.draw.rect(self.screen,(255,255,255),(j,i,j+100,i+100),0)
                # rectangles with the colors defined in the list colors
                pygame.draw.rect(self.screen,self.colors[c],(j+3,i+3,j+95,i+95),0)
                c+=1

    # Function to make the agent move (pink ball)
    def select_action(self, current_state):
        # create an empty list to store the future actions
        possible_actions = []

        # if a random number is <= to epsilon you can move
        if np.random.uniform() <= self.epsilon:
            if self.current_pos[1] != 0:
                possible_actions.append("left")
            if self.current_pos[1] != self.n-1:
                possible_actions.append("right")
            if self.current_pos[0] != 0:
                possible_actions.append("up")
            if self.current_pos[0] != self.n-1:
                possible_actions.append("down")
            action = self.actions[possible_actions[r(0,len(possible_actions) - 1)]]
        # if not, you use the Q table
        else:
            m = np.min(self.Q[current_state])
            #  curent state and the action (up)
            if self.current_pos[0] != 0:
                possible_actions.append(self.Q[current_state,0])
            else:
                possible_actions.append(m - 100)

            #  curent state and the action (down)
            if self.current_pos[0] != self.n-1:
                possible_actions.append(self.Q[current_state,1])
            else:
                possible_actions.append(m - 100)

            #  curent state and the action (left)
            if self.current_pos[1] != 0:
                possible_actions.append(self.Q[current_state,2])
            else:
                possible_actions.append(m - 100)

            #  curent state and the action (right)
            if self.current_pos[1] != self.n-1:
                possible_actions.append(self.Q[current_state,3])
            else:
                possible_actions.append(m - 100)

            # new action
            action = random.choice([i for i, a in enumerate(possible_actions) if a == max(possible_actions)])
            return action

    # Function to apply the reinforcement learning
    def episode(self):
        # create a terminal list with 16 values ranging from 0 to 15.
        self.terminals.append(self.n**2 - 1)

        # creation of the initial random states.
        states = {}
        k = 0
        for i in range(self.n):
            for j in range(self.n):
                states[(i,j)] = k
                k+=1

        self.current_state = states[(self.current_pos[0], self.current_pos[1])]

        # create the action
        action = self.select_action(self.current_state)
        if action == 0:
            self.current_pos[0] -= 1
        elif action == 1:
            self.current_pos[0] += 1
        elif action == 2:
            self.current_pos[1] -= 1
        elif action == 3:
            self.current_pos[1] += 1
        # create the new state by inserting the current position as a tuple in the randomly created state before
        new_state = states[(self.current_pos[0],self.current_pos[1])]

        # update the q-table with new q-values if they don't exist.
        if new_state not in self.terminals:
            # The core of the algorithm is a Bellman equation
            # as a simple value iteration update, using the weighted average of the current value and the new information
            # Q-Learning goal is to select the action with highest value at a state to move to another state
            # The agent in state, will pick a random action and receive reward(s) and updates
            # the value of action in state according to the equation.
            self.Q[self.current_state,action] += self.alpha*(self.reward[self.current_pos[0],self.current_pos[1]] + self.gamma*(np.max(self.Q[new_state])) - self.Q[self.current_state,action])
        else:
            self.Q[self.current_state,action] += self.alpha*(self.reward[self.current_pos[0],self.current_pos[1]] - self.Q[self.current_state,action])
            self.current_pos = [0,0]
            self.epsilon -= 1e-3


# call the class by defining:
# the size of the cell, the number of walls, the learning rate, the discount, the epsilon
# the reward/penalty for the treasure, the walls penalty, the exit reward
# you can change these values as you like.
dm = dungeon_map(8, 4, 0.2, 0.9, 0.1, 0, -10, 10)


# make the program run in an infinite loop
# I noticed that the 1st iteration is often not good with the start or the exit being blocked.
# Restart a new one.
run = True
while run:
    sleep(0.1)
    dm.screen.fill(dm.background)
    dm.layout()
    # the agent is a pink circle
    pygame.draw.circle(dm.screen,(230,25,129),(dm.current_pos[1]*100 + 50,dm.current_pos[0]*100 + 50),30,0)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    pygame.display.flip()
    dm.episode()
pygame.quit()
