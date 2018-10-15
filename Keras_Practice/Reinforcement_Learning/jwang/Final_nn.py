import gym, random
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from collections import deque


class ReinforcementModel:
    def __init__(self, env):
        self.lr = .001
        self.gamma = 0.9
        self.random_precent = 1
        self.Decay = .995
        self.Min = .01
        self.memory = deque(maxlen = 2000)
        self.env = env
        self.model = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim = self.env.observation_space.shape[0], activation = 'relu'))
        model.add(Dense(48, activation = 'relu'))
        model.add(Dense(24, activation = 'relu'))
        model.add(Dense(self.env.action_space.n)) # what choice should it make

        optimizer = Adam(lr = self.lr)
    
        model.compile(loss = 'mse', optimizer = optimizer)

        return model
    
    def memAdd(self, state, action, reward, newState, done):
        self.memory.append([state, action, reward, newState, done])


    def act(self, state):
        self.random_precent *= self.Decay
        self.random_precent = max(self.Min, self.random_precent)
        if np.random.random() < self.random_precent:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])

    def replay(self):
        batchSample = 32
        if len(self.memory) < batchSample:
            return
        samples = random.sample(self.memory, batchSample)
        for sample in samples:
            state, action, reward, nextState, done = sample
            target = self.model.predict(state)
            qupdate = reward
            if not done:
                qupdate = reward + max(self.model.predict(nextState)[0]) * self.gamma
            qPred = self.model.predict(state)
            qPred[0][action] = qupdate
            self.model.fit(state, qPred, epochs = 1, verbose = 0)



if __name__ == '__main__':

    env = gym.make('CartPole-v0')
    training_round = 100000
    training_length = 400




    
    dqn = ReinforcementModel(env)

#print(env.action_space.n)
#print(env.observation_space.low)


    for i_episode in range(training_round):
        observation = env.reset().reshape(1, 4)
        for t in range(training_length):
            action = dqn.act(observation)
            env.render()
            #            print(observation)
            
            newState, reward, done, info = env.step(action)
            if done:
                reward = -reward
            newState = newState.reshape(1, 4)
            dqn.memAdd(observation, action, reward, newState, done)
            dqn.replay()
            observation = newState
            if done:
                break
        if t < 199:
            print(str(i_episode) + ' round, Failed to complete trial with step: ' + str(t))
        else:
            print(str(i_episode) + 'Success')

