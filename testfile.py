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
from keras.layers import Dense, Flatten
from collections import deque

# For calcuations
import numpy as np
import random
import cv2

# Load the game
# scenarion = 'scenario.json'
env = retro.make(game='SuperMarioBros-Nes') # scenario = scenarion


def epsilon_greedy(q_values, epsilon): # Don't need to predict every time. but shit the same
    policy = np.zeros(q_values.shape[1])
    if random.uniform(0,1) <= epsilon:
        policy[np.argmax(q_values)] = 1
    else:
        for i in range(0, len(policy)):
            policy[i] = 1 / len(policy)
    return policy

# Actions
actions = [('RIGHT'), ('A'), ('RIGHT', 'A'), ('RIGHT', 'B')]
action_dict = {():[0,0,0,0,0,0,0,0,0] 
        , ('LEFT'): [0,0,0,0,0,0,1,0,0]
        , ('RIGHT'): [0,0,0,0,0,0,0,1,0]
        , ('A') : [0,0,0,0,0,0,0,0,1]
        , ('B'): [1,0,0,0,0,0,0,0,0]
        , ('LEFT','A'): [0,0,0,0,0,0,1,0,1]
        , ('RIGHT', 'A'): [0,0,0,0,0,0,0,1,1]
        , ('LEFT','B'): [1,0,0,0,0,0,1,0,0]
        , ('RIGHT', 'B'): [1,0,0,0,0,0,0,1,0]
        , ('RIGHT', 'A', 'B'): [1,0,0,0,0,0,0,1,1]
        , ('DOWN'): [0,0,0,0,0,1,0,0,0]
        }

# Parameters
output_size = len(actions)
epsilon = 0.99
gamma = 0.9
epsilon_decay = 0.9
mini_batches = 128
observation_steps = 40000
done = False
action_array = np.zeros(output_size)

queue = deque() # To save states for learning

# The neural network
model = Sequential()
model.add(Dense(20, input_shape=(2,) + (16,12,1), init='uniform', activation='relu')) # 
model.add(Flatten())       # Flatten input so as to have no problems with processing
model.add(Dense(18, init='uniform', activation='relu'))
model.add(Dense(10, init='uniform', activation='relu'))
model.add(Dense(output_size, init='uniform', activation='linear'))    # Same number of outputs as possible actions

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

while True:
    # Expolre the environment
    observation = env.reset()
    observation2 = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY) # Make gray and resize to get fewer inputs.
    observation = cv2.resize(observation2, (16,12)).reshape(16,12,-1)
    obs = np.expand_dims(observation, axis=0) # Is this needed if we go grayScale?
    state = np.stack((obs, obs), axis=1)
    total_reward = 0

    for i in range(observation_steps):
        if i % 4 == 0: # Every fourth frame/step take a new action
            # action_array = np.zeros(output_size) # Me no like this one

            Q = model.predict(state) # Get state from model/network
            policy = epsilon_greedy(Q, epsilon) # Get greedy or random choice for observation
            action = actions[np.argmax(policy)]
            # action = np.random.choice(actions, p = policy) # Choose an action

            action = action_dict[action] 
            observation_new, reward, done, info = env.step(action)
            observation_new2 = cv2.cvtColor(observation_new, cv2.COLOR_BGR2GRAY)
            observation_new = cv2.resize(observation_new2, (16,12)).reshape(16,12,-1)

            obs_new = np.expand_dims(observation_new, axis=0)
            state_new = np.append(np.expand_dims(obs_new, axis=0), state[:, :1, :], axis=1)
            queue.append((state, action, reward, state_new, done)) # To remember
            state = state_new
            total_reward += reward
            env.render() 
        if done:
            env.reset()
            obs = np.expand_dims(observation, axis=0)
            state = np.stack((obs, obs), axis=1)
    print('Observing Finished')
    print('Total reward in exploration: {}'.format(total_reward))

    env.reset()
    #obs = np.expand_dims(observation, axis=0)
    #state = np.stack((obs, obs), axis=1)

    # SECOND STEP: Learning from the observations (Experience replay)
    minibatch = random.sample(queue, mini_batches)                              # Sample some moves

    inputs_shape = (mini_batches,) + state.shape[1:]
    inputs = np.zeros(inputs_shape)
    targets = np.zeros((mini_batches, output_size))

    for i in range(0, mini_batches):
        state = minibatch[i][0]
        action = minibatch[i][1]
        reward = minibatch[i][2]
        state_new = minibatch[i][3]
        done = minibatch[i][4]
        
    # Build Bellman equation for the Q function
        inputs[i:i+1] = np.expand_dims(state, axis=0)
        targets[i] = model.predict(state)
        Q_sa = model.predict(state_new)
        
        if done:
            targets[i, action] = reward
        else:
            targets[i, action] = reward + gamma * np.max(Q_sa)

    # Train network to output the Q function
        model.train_on_batch(inputs, targets)
    print('Learning Finished')
    epsilon = epsilon * epsilon_decay
'''
# THIRD STEP: Play!
observation = env.reset()
obs = np.expand_dims(observation, axis=0)
state = np.stack((obs, obs), axis=1)
done = False
tot_reward = 0.0
while not done:
    t += 1

    action_array = np.zeros(env.action_space.n)
    Q = model.predict(state)        
        
#    policy = epsilon_greedy(Q, epsilon)
#    action = np.random.choice(env.action_space.n, policy)         
    action = np.argmax(Q)
    action_array[action] = 1
    observation, reward, done, info = env.step(action_array)
    obs = np.expand_dims(observation, axis=0)
    state = np.append(np.expand_dims(obs, axis=0), state[:, :1, :], axis=1)    
    tot_reward += reward
    if t % 100 == 0:
        env.render()
        print('Total reward: {}'.format(tot_reward))
        print(Q)
print('Game ended! Total reward: {}'.format(reward))
'''