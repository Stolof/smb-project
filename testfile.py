'''
TODO
-----
- Could add parser for starting at different levels, and perhaps switch games

--------- TORSDAG
- CNN
- Plot values. Calc loss etc.
- queue size 3000, 300 states/ run,  with 128 batch_size. (De tar 64 av 1000)
- Double networks DDQN
- How many frames should we stack, 4?

- td_targets is not working
- Make a plot of good info
'''
# The gym
import retro
# To make the neural network
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras import regularizers
from collections import deque

# For calcuations
import numpy as np
import random
import cv2

def calculate_td_targets(q1_batch, q2_batch, r_batch, t_batch, gamma=.90):
    '''
    Calculates the TD-target used for the loss
    : param q1_batch: Batch of Q(s', a) from online network, shape (N, num actions)
    : param q2_batch: Batch of Q(s', a) from target network, shape (N, num actions)
    : param r_batch: Batch of rewards, shape (N, 1)
    : param t_batch: Batch of booleans indicating if state, s' is terminal, shape (N, 1)
    : return: TD-target, shape (N, 1)
    '''
    Y = np.zeros(r_batch.shape)
    actions = np.zeros(r_batch.shape)

    for i in range(0, len(r_batch)):
        if t_batch[i]:
            Y[i] = r_batch[i]
            actions[i] = 0
        else:
            Y[i] = r_batch[i] + gamma * (q2_batch[i][0][np.argmax(q1_batch[i][0])])
            actions[i] = np.argmax(q2_batch[i][0]) # Is this one correct!??!?!
    return Y, actions

def epsilon_greedy(q_values, epsilon): # Don't need to predict every time. but shit the same
    policy = np.zeros(len(q_values))
    if random.uniform(0,1) >= epsilon:
        policy[np.argmax(q_values)] = 1
    else:
        for i in range(0, len(policy)):
            policy[i] = 1 / len(policy)
    return policy

def reframe(obs, height, width): # Also crop away the top input 256x240
    obs = obs[32:, :]
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY) # Make gray and resize to get fewer inputs.
    obs = cv2.resize(obs, (height,width)) #.reshape(height,width,-1)
    return obs

def make_nn(): # The neural network
    model = Sequential()
    model.add(Dense(20, input_shape=(2,) + (img_height,img_width,1), init='uniform', activation='relu')) # input should be every pixel 16*12, how many frames maybe 4
    model.add(Flatten())       # Flatten input so as to have no problems with processing
    model.add(Dense(18, init='uniform', activation='relu'))
    model.add(Dense(10, init='uniform', activation='relu'))
    model.add(Dense(output_size, init='uniform', activation='linear'))    # Same number of outputs as possible actions

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    model.summary()
    return model

def make_cnn(): # Master CNN, no regularization.
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(8, 8), input_shape= (img_height,img_width) + (2,) , activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(256, init='uniform', activation='relu'))    # Same number of outputs as possible actions
    model.add(Dense(output_size, init='uniform', activation='linear'))    # Same number of outputs as possible actions
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    model.summary()
    return model

def make_state(): # Remove env.reset and remake.
    observation = env.reset()
    observation = reframe(observation, img_height, img_width)
    state = np.stack((observation, observation), axis=2)
    return state

def switch_networks(): # Global values
    offline_parameters = offline_network.get_weights()
    online_parameters = online_network.get_weights()
    online_network.set_weights(offline_parameters)
    offline_network.set_weights(online_parameters)

# Load the game
env = retro.make(game='SuperMarioBros-Nes') # scenario = 'scenario.json'

# Actions
actions = [('RIGHT'), ('RIGHT', 'A')]
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
epsilon_decay = 0.005 # Decay for epsilon
mini_batches = 128 # Mini batch for learning, 300 states per exploration, 128 states is too much?? 
img_height = 64 # Rescale size 256x240 /16... 84x84 needed?
img_width = 64
done = False
epsilon_goal = 0.1 # When the learning phace should stop
action_array = np.zeros(output_size) # No need for this shit
frames_per_action = 10
exploration_runs = 1 # One to not run first run.

queue = deque(maxlen=3000) # To save states for learning

# model = make_nn()
online_network = make_cnn()
offline_network = make_cnn()

while True:
    state = make_state()

    total_reward = 0
    t = 0
    
    # Expolre the environment
    while not (done): 
        if t % frames_per_action == 0: # Every X frame, step take a new action 

            action_array = np.arange(output_size)
            Q = online_network.predict(np.expand_dims(state, axis=0)) # Get state from model/network
            policy = epsilon_greedy(Q[0], epsilon) # Get greedy or random choice for observation
            action = np.random.choice(action_array, p = policy) # Choose an action
            action = actions[action]

            action = action_dict[action]
            observation_new, reward, done, info = env.step(action)
            observation_new = reframe(observation_new, img_height, img_width)
            observation_new = np.expand_dims(observation_new, axis=-1)

            state_new = np.append(observation_new, state[:, :, :1], axis=2)
            queue.append((state, action, reward, state_new, done)) # To remember

            state = state_new
            total_reward += reward
            env.render()
        else: # Check if dead and update reward every step. 
            _, reward, done, _ = env.step(action)
            total_reward += reward
        t += 1
    print('Exploration Finished')
    print('Total reward in exploration: {}'.format(total_reward))

    state = make_state()

    # SECOND STEP: Learning from the observations (Experience replay)
    minibatch = random.sample(queue, mini_batches)                              # Sample some moves

    inputs_shape = (mini_batches,) + state.shape
    inputs = np.zeros(inputs_shape)
    targets = np.zeros((mini_batches, output_size))

    reward_batch = np.zeros(mini_batches)
    done_batch = np.zeros(mini_batches)
    Q_sa_1 = [] # np.zeros(mini_batches)
    Q_sa_2 = [] # np.zeros(mini_batches)

    for i in range(0, mini_batches): # Remove this shit
        state = minibatch[i][0]
        action = minibatch[i][1]
        reward = minibatch[i][2]
        state_new = minibatch[i][3]
        done = minibatch[i][4]
        
    # Build Bellman equation for the Q function
        state = np.expand_dims(state, axis=0)
        state_new = np.expand_dims(state_new, axis=0)
        inputs[i:+1] = state
        targets[i] = online_network.predict(state)
        reward_batch[i] = reward
        done_batch[i] = done
        Q_sa_1.append(online_network.predict(state_new)) 
        Q_sa_2.append(offline_network.predict(state_new))

    td_target, actions_to_train = calculate_td_targets(Q_sa_1, Q_sa_2, reward_batch, done_batch, gamma)
    for i in range(0,mini_batches):
        targets[i, int(actions_to_train[i])] = td_target[i] 
    online_network.train_on_batch(inputs, targets)

    if random.uniform(0,1) > 0.5:
        switch_networks()

    print('Learning Finished')
    print('Epsilon is {}'.format(epsilon))
    if(epsilon > epsilon_goal):
       epsilon = epsilon - epsilon_decay

    if(exploration_runs % 50 == 0): # THIRD STEP: Play every 50 step
        model_name = 'play_state_DDQN2.h5'
        online_network.save_weights(model_name)
        print('Saved the model as {}'.format(model_name))
        # model = model.load_weights('play_state.h5')

        state = make_state()
        done = False
        total_reward = 0
        t = 0

        while not done:
            if t % 10 == 0:
                Q = online_network.predict(np.expand_dims(state, axis=0)) # Model chooses action
                action = actions[np.argmax(Q[0])]
                action = action_dict[action]

                observation, reward, done, info = env.step(action)
                observation = reframe(observation, img_height, img_width)
                observation = np.expand_dims(observation, axis=-1)
                state = np.append(observation, state[:, :, :1], axis=2)
        
                env.render()
            else:
                _, reward, done, _ = env.step(action)
            t += 1
            total_reward += reward
        print('Game ended! Total reward: {}'.format(total_reward))
    exploration_runs += 1 
