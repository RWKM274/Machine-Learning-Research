import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from collections import deque

# modified https://github.com/pdemange/Machine-Learning-Research/blob/master/Keras_Practice/Reinforcement_Learning/pdemange/pdemange_cartpole.py
class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=2000)

        self.learning_rate = .0001
        self.gamma = .9
        self.epsilon = 1.0
        self.epsilon_decay = .995
        self.epsilon_min = .01
        self.tau = .125

        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()
        state_shape = self.env.observation_space.shape
        model.add(Dense(24, input_dim=state_shape[0], activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                target[0][action] = reward + max(self.target_model.predict(new_state)[0]) * self.gamma

            self.model.fit(state, target, epochs=1, verbose=0)


if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    trials = 100000
    trial_len = 400

    dqn = DQN(env=env)
    steps = []

    for trial in range(trials):
        cur_state = env.reset().reshape(1, 4)
        for step in range(trial_len):
            action = dqn.act(cur_state)
            env.render()
            new_state, reward, done, _ = env.step(action)
            if done:
                reward = -reward
            new_state = new_state.reshape(1, 4)
            dqn.remember(cur_state, action, reward, new_state, done)

            dqn.replay()  # internally iterates default (prediction) model

            cur_state = new_state
            if done:
                break
        if step <= 199:
            print('rounds = ' + str(trial) + ':' + 'Failed to complete in ' + str(step) + ' steps')
        else:
            print('rounds = ' + str(trial) + ':' + 'Completed in trials' + str(step) + ' steps')