#pip install pygame
import numpy as np
import pygame
from time import time,sleep
from random import randint as r
import random
import pickle


class dungeon_map:
    def __init__(self, size, walls, learning_rate, discount, epsilon):
        self.n = size
        self.background = (51,51,51)
        self.screen = pygame.display.set_mode((self.n*100,self.n*100))
        self.penalities = walls
        # learning rate
        self.alpha = learning_rate
        # weight measure of how important do we find future actions vs current action or future reward over current reward. values between 0 and 1
        self.gamma = discount
        # values between 0 and 1. The higher the epsilon, the more likely we are to perform a random action
        # overtime, we want to stop our model to stop exploring
        self.epsilon = epsilon
        self.actions = {"up": 0,"down" : 1,"left" : 2,"right" : 3}
        self.current_pos = [0,0]
        self.Q = np.zeros((self.n**2,4))
        self.colors =  [(51,51,51) for i in range(self.n**2)]
        self.terminals = []
        self.reward = np.zeros((self.n,self.n))

    def layout(self):
        self.terminals.append(self.n**2 - 1)
        # exit, I put the reward for the exit to be the highest one.
        self.reward[self.n-1,self.n-1] = 2
        #treasure, I put -1 for the treasure for the ball to pass by the value, but not stay there.
        self.reward[2,1] = -1

        # start
        self.colors[0] = (0,255,0)
        # exit
        self.colors[self.n**2 - 1] = (0,255,0)
        #treasure
        self.colors[self.n**2 - 7] = (253,255,71)

        while self.penalities != 0:
            i = r(0,self.n-1)
            j = r(0,self.n-1)
            # if the case in the reward array are equal to 0, we replace it by -5 and put penalties
            if self.reward[i,j] == 0 and [i,j] != [0,0] and [i,j] != [self.n-1,self.n-1] and [i,j]!=[1,2]:
                self.reward[i,j] = -5
                self.penalities -= 1
                self.colors[self.n*i+j] = (255,0,0)  #red
                self.terminals.append(self.n*i+j)
        c = 0
        for i in range(0,self.n*100,100):
            for j in range(0,self.n*100,100):
                pygame.draw.rect(self.screen,(255,255,255),(j,i,j+100,i+100),0)
                pygame.draw.rect(self.screen,self.colors[c],(j+3,i+3,j+95,i+95),0)
                c+=1


    def select_action(self, current_state):
        # initial q_table
        possible_actions = []
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
        else:
            m = np.min(self.Q[current_state])
            if self.current_pos[0] != 0:
                possible_actions.append(self.Q[current_state,0])
            else:
                possible_actions.append(m - 100)
            if self.current_pos[0] != self.n-1:
                possible_actions.append(self.Q[current_state,1])
            else:
                possible_actions.append(m - 100)
            if self.current_pos[1] != 0:
                possible_actions.append(self.Q[current_state,2])
            else:
                possible_actions.append(m - 100)
            if self.current_pos[1] != self.n-1:
                possible_actions.append(self.Q[current_state,3])
            else:
                possible_actions.append(m - 100)
            # action = np.argmax(possible_actions)
            action = random.choice([i for i,a in enumerate(possible_actions) if a == max(possible_actions)])
            return action

    def episode(self):
        global current_state
        states = {}
        k = 0
        for i in range(self.n):
            for j in range(self.n):
                states[(i,j)] = k
                k+=1
        current_state = states[(self.current_pos[0],self.current_pos[1])]
        action = self.select_action(current_state)
        if action == 0:
            self.current_pos[0] -= 1
        elif action == 1:
            self.current_pos[0] += 1
        elif action == 2:
            self.current_pos[1] -= 1
        elif action == 3:
            self.current_pos[1] += 1
        new_state = states[(self.current_pos[0],self.current_pos[1])]
        if new_state not in self.terminals:
            # The core of the algorithm is a Bellman equation
            # as a simple value iteration update, using the weighted average of the current value and the new information
            self.Q[current_state,action] += self.alpha*(self.reward[self.current_pos[0],self.current_pos[1]] + self.gamma*(np.max(self.Q[new_state])) - self.Q[current_state,action])
        else:
            self.Q[current_state,action] += self.alpha*(self.reward[self.current_pos[0],self.current_pos[1]] - self.Q[current_state,action])
            self.current_pos = [0,0]
            self.epsilon -= 1e-3


dm = dungeon_map(4, 4, 0.01, 0.9, 0.25)
run = True
for i in range(000):
    dm.episode()
while run:
    sleep(0.1)
    dm.screen.fill(dm.background)
    dm.layout()
    pygame.draw.circle(dm.screen,(230,25,129),(dm.current_pos[1]*100 + 50,dm.current_pos[0]*100 + 50),30,0)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    pygame.display.flip()
    dm.episode()

pygame.quit()
#print(epsilon)
# list for accuracies
#episode_rewards= []

# moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode="valid")
# plt.plot([i for i in range(len(moving_avg))], moving_avg)
# plt.ylabel(f"reward {SHOW_EVERY} ma")
# plt.xlabel("episode #")
# plt.show()

# f = open("Q.txt","w")
# f.write(pickle.dumps(Q))
# f.close()
