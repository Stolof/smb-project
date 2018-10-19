import retro
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from collections import deque


class Agent:
    def __init__(self):
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.95

        self.memory = deque(maxlen=100000)

        self.model = Sequential()
        self.model.add(Dense(16, input_dim=9, activation='relu'))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(10, activation='linear'))
        self.model.compile(loss='mse', optimizer='nadam')
        self.model.summary()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if self.epsilon > np.random.uniform():
            return np.random.randint(10)
        expected_returns = self.model.predict(state)[0]
        print("Expected return: {}".format(np.max(expected_returns)))
        return np.argmax(self.model.predict(state)[0])

    def replay(self):
        mini_batch = random.sample(self.memory, 1024)
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + 0.9 * \
                         np.max(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0, action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == '__main__':
    actions = [(),('LEFT'), ('RIGHT'), ('A'), ('B'), ('DOWN'), ('LEFT','A'), ('RIGHT', 'A'), ('LEFT', 'B'), ('RIGHT', 'B')]
    action_dict = {():[0,0,0,0,0,0,0,0,0] 
        , ('LEFT'): [0,0,0,0,0,0,1,0,0]
        , ('RIGHT'): [0,0,0,0,0,0,0,1,0]
        , ('A') : [0,0,0,0,0,0,0,0,1]
        , ('B'): [1,0,0,0,0,0,0,0,0]
        , ('LEFT','A'): [0,0,0,0,0,0,1,0,1]
        , ('RIGHT', 'A'): [0,0,0,0,0,0,0,1,1]
        , ('LEFT','B'): [1,0,0,0,0,0,1,0,0]
        , ('RIGHT', 'B'): [1,0,0,0,0,0,0,1,0]
        , ('DOWN'): [0,0,0,0,0,1,0,0,0]}

    env = retro.make('SuperMarioBros-Nes', retro.State.DEFAULT)
    agent = Agent()

    for e in range(1000):
        env.reset()
        action = [0,0,0,0,0,0,0,0,0]
        _, _, done, state = env.step(action)
        state = np.reshape(list(state.values()), [1, 9])
        t = 0
        
        while not done:
            if t % 4 == 0:
                action_number = agent.act(state)
                action = action_dict[actions[action_number]]
            if t % 100 == 0:
                env.render()
            _, reward, done, next_state = env.step(action)
            print(next_state )
            next_state = np.reshape(list(next_state.values()), [1, 9])

            agent.remember(state, action_number, reward, next_state, done)

            state = next_state
            t += 1
            
        

        print('Learning... epsilon = {}'.format(agent.epsilon))
        agent.replay()
