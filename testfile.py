'''
TODO
-----
Could add parser for starting at different levels, and perhaps switch games

This model gets rewards from
Coins +1
Stars +3
Mushroom
...

'''
# The gym
import retro

# To make the neural network
from keras.models import Sequential
from keras.layers import Dense

# For calcuations
import numpy as np
import random

# Load the game
env = retro.make(game='SuperMarioBros-Nes')

# The neural network
model = Sequential()


# Run the game and agent
while True:
    ob = env.reset()
    env.render()
