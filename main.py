import retro
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from collections import deque
from model import CNN
from retro_wrappers import make_retro, wrap_deepmind_retro

GAMMA = 0.99
MEMORY_SIZE = 900000
BATCH_SIZE = 32
TRAINING_FREQUENCY = 1000
OFFLINE_NETWORK_UPDATE_FREQUENCY = 40000
# MODEL_PERSISTENCE_UPDATE_FREQUENCY = 10000
REPLAY_START_SIZE = 50000

EPSILON_MAX = 1.0
EPSILON_MIN = 0.1
EXPLORATION_STEPS = 850000
EPSILON_DECAY = (EPSILON_MAX-EPSILON_MIN)/EXPLORATION_STEPS

class Agent:
    def __init__(self):
        self.epsilon = EPSILON_MAX
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.offline = CNN((4, 84, 84), 6)
        self.online = CNN((4, 84, 84), 6)

    def remember(self, current_state, action, reward, next_state, done):
        self.memory.append((current_state, action, reward, next_state, done))

    def act(self, current_state):
        if self.epsilon > np.random.uniform():
            return np.random.randint(6)
        current_state = np.swapaxes(np.expand_dims(current_state, axis=0), 1, 3)
        expected_returns = self.online.model.predict(current_state)[0]
        return np.argmax(expected_returns)

    def replay(self):
        mini_batch = random.sample(self.memory, BATCH_SIZE)
        for current_state, action, reward, next_state, done in mini_batch:
            current_state = np.swapaxes(np.expand_dims(current_state, axis=0), 1, 3)
            next_state = np.swapaxes(np.expand_dims(next_state, axis=0), 1, 3)

            target = reward
            if not done:
                target = reward + GAMMA * \
                         np.max(self.offline.model.predict(next_state)[0])
            target_f = self.online.model.predict(current_state)
            target_f[0, action] = target
            self.online.model.fit(current_state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon -= EPSILON_DECAY

if __name__ == '__main__':
    actions = [(), ('LEFT'), ('RIGHT'), ('A'), ('RIGHT', 'A'), ('LEFT','A')]
    action_dict = {():[0,0,0,0,0,0,0,0,0] 
        , ('LEFT'): [0,0,0,0,0,0,1,0,0]
        , ('RIGHT'): [0,0,0,0,0,0,0,1,0]
        , ('A') : [0,0,0,0,0,0,0,0,1]
        , ('LEFT','A'): [0,0,0,0,0,0,1,0,1]
        , ('RIGHT', 'A'): [0,0,0,0,0,0,0,1,1]}

    env = wrap_deepmind_retro(retro.make('SuperMarioBros-Nes', retro.State.DEFAULT))
    agent = Agent()

    total_step = 0

    for e in range(1000):
        current_state = env.reset()
        step = 0
        total_reward = 0
        done = False
        
        while not done:
            if total_step % 100 == 0:
                env.render()
            action_number = agent.act(current_state)
            action = action_dict[actions[action_number]]
            next_state, reward, done, info = env.step(action)

            agent.remember(current_state, action_number, reward, next_state, done)

            current_state = next_state
            total_reward += reward
            total_step += 1

            if total_step > REPLAY_START_SIZE and total_step % TRAINING_FREQUENCY == 0:
                #print('Learning... epsilon = {}'.format(agent.epsilon))
                agent.replay()
            
            if total_step % OFFLINE_NETWORK_UPDATE_FREQUENCY == 0:
                print('Updating offline network...')
                agent.offline.model.set_weights(agent.online.model.get_weights())

        print('Total reward = {}'.format(total_reward))

'''
--------------------
        batch = np.asarray(random.sample(self.memory, BATCH_SIZE))
        if len(batch) < BATCH_SIZE:
            return

        current_states = []
        q_values = []
        max_q_values = []

        for entry in batch:
            current_state = np.expand_dims(np.asarray(entry["current_state"]).astype(np.float64), axis=0)
            current_states.append(current_state)
            next_state = np.expand_dims(np.asarray(entry["next_state"]).astype(np.float64), axis=0)
            next_state_prediction = self.ddqn_target.predict(next_state).ravel()
            next_q_value = np.max(next_state_prediction)
            q = list(self.ddqn.predict(current_state)[0])
            if entry["terminal"]:
                q[entry["action"]] = entry["reward"]
            else:
                q[entry["action"]] = entry["reward"] + GAMMA * next_q_value
            q_values.append(q)
            max_q_values.append(np.max(q))

        fit = self.ddqn.fit(np.asarray(current_states).squeeze(),
                            np.asarray(q_values).squeeze(),
                            batch_size=BATCH_SIZE,
                            verbose=0
--------------------
'''