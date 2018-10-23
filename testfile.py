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
from keras.layers import Dense, Flatten, Conv2D
from collections import deque

# For calcuations
import numpy as np
import random
import cv2

# Load the game
env = retro.make(game='SuperMarioBros-Nes') # scenario = 'scenario.json'


def epsilon_greedy(q_values, epsilon): # Don't need to predict every time. but shit the same
    policy = np.zeros(q_values.shape[1])
    if random.uniform(0,1) <= epsilon:
        policy[np.argmax(q_values)] = 1
    else:
        for i in range(0, len(policy)):
            policy[i] = 1 / len(policy)
    return policy

def reframe(obs, height, width):
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY) # Make gray and resize to get fewer inputs.
    obs = cv2.resize(obs, (height,width)).reshape(height,width,-1)
    return obs

# Actions
actions = [('RIGHT'), ('RIGHT', 'A'), ('RIGHT', 'B'), ('RIGHT', 'A', 'B')]
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
epsilon = 1 # Chanche of taking a random action 
gamma = 0.90 # Decay of reward for every step. Care about the future?
epsilon_decay = 0.98 # Decay for epsilon
mini_batches = 128 # Mini batch for learning 
observation_steps = 10000 # 10k almost 400 time.
img_height = 16 # Rescale size
img_width = 12 
done = False
epsilon_goal = 0.1 # When the learning phace should stop
action_array = np.zeros(output_size) # No need for this shit

queue = deque(maxlen=100000) # To save states for learning

# The neural network
model = Sequential()
model.add(Dense(20, input_shape=(2,) + (img_height,img_width,1), init='uniform', activation='relu')) # 
model.add(Flatten())       # Flatten input so as to have no problems with processing
model.add(Dense(18, init='uniform', activation='relu'))
model.add(Dense(10, init='uniform', activation='relu'))
model.add(Dense(output_size, init='uniform', activation='linear'))    # Same number of outputs as possible actions

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.summary()

while True:
    observation = env.reset()
    observation = reframe(observation, img_height, img_width)
    obs = np.expand_dims(observation, axis=0) # Is this needed if we go grayScale?
    state = np.stack((obs, obs), axis=1)

    total_reward = 0
    t = 0
    
    # Expolre the environment
    while not (t == observation_steps): # It does not notice when it's game over?
        if t % 10 == 0: # Every fourth frame/step take a new action / reward should be 

            action_array = np.arange(output_size)
            Q = model.predict(state) # Get state from model/network
            policy = epsilon_greedy(Q, epsilon) # Get greedy or random choice for observation
            action = np.random.choice(action_array, p = policy) # Choose an action
            action = actions[action]

            action = action_dict[action]
            observation_new, reward, done, info = env.step(action)
            observation_new = reframe(observation_new, img_height, img_width)
            #print('Info {}'.format(info))
            #print('Reward {}'.format(reward))
            obs_new = np.expand_dims(observation_new, axis=0)

            state_new = np.append(np.expand_dims(obs_new, axis=0), state[:, :1, :], axis=1)
            queue.append((state, action, reward, state_new, done)) # To remember

            state = state_new
            total_reward += reward
            env.render()
        else: # Check if dead every other step. Should update reward aswell? 
            _, reward, done, info = env.step(action)
            total_reward += reward
        t += 1
        if done: # Observations 
            print(' You are done!')
            break
    print('Observing Finished')
    print('Total reward in exploration: {}'.format(total_reward, done))

    # This is only for the last step??
    observation = env.reset() # No need if the loop gets done..
    observation = reframe(observation, img_height, img_width)
    obs = np.expand_dims(observation, axis=0)
    state = np.stack((obs, obs), axis=1)

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
    print('Epsilon is {}'.format(epsilon))
    if (epsilon < epsilon_goal):
        break

# THIRD STEP: Play!
observation = env.reset()
observation = reframe(observation, img_height, img_width)
obs = np.expand_dims(observation, axis=0)
state = np.stack((obs, obs), axis=1)

done = False
total_reward = 0
t = 0

while not done:
    t += 1

    Q = model.predict(state) # Model chooses action
    action = actions[np.argmax(Q)]
    action = action_dict[action]
    observation, reward, done, info = env.step(action)

    observation = reframe(observation, img_height, img_width)
    obs = np.expand_dims(observation, axis=0)
    state = np.append(np.expand_dims(obs, axis=0), state[:, :1, :], axis=1)    
    
    tot_reward += reward
    if t % 100 == 0:
        env.render()
print('Game ended! Total reward: {}'.format(reward))