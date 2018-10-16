import gym, random
from gym import envs
import numpy as np
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from collections import deque


class ReinforcementModel:
    def __init__(self, env):
        self.lr = .0001
        self.gamma = 0.9
        self.random_precent = 1
        self.Decay = .995
        self.Min = .01
        self.memory = deque(maxlen = 2000)
        self.env = env
        self.model = self.create_model()

    def create_model(self):
        model = Sequential()
        '''
        model.add(Conv2D(32, (3, 3), input_shape = (210, 160, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))

        model.add(Flatten())
        '''
        model.add(Dense(48, activation = 'relu', input_dim = self.env.observation_space.shape[0]))
        model.add(Dense(96, activation = 'relu'))
        model.add(Dense(48, activation = 'relu'))
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
        return np.argmax(self.model.predict(state))

    def replay(self):
        batchSample = 32
        if len(self.memory) < batchSample:
            return
        samples = random.sample(self.memory, batchSample)
        for sample in samples:
            state, action, reward, nextState, done = sample
            #target = self.model.predict(state)
            qupdate = reward
            #print(nextState, state)
            if(nextState[0][14] - state[0][14] > 0):
                print("Win")
                qupdate = 10 + max(self.model.predict(nextState)[0]) * self.gamma
            elif(nextState[0][18] > 0 and nextState[0][18] < 255):
                print("Hit")
                qupdate = 5 + max(self.model.predict(nextState)[0]) * self.gamma
            elif(nextState[0][13] - state[0][13] > 0):
                print("Lost")
                qupdate = -10
            else:
                qupdate = 0
            qPred = self.model.predict(state)
            qPred[0][action] = qupdate
            self.model.fit(state, qPred, epochs = 1, verbose = 0)



if __name__ == '__main__':
    #for i in envs.registry.all():
        #print(i)
    #print(envs.registry.all())
    #env = gym.make('CartPole-v0')

    env = gym.make('Pong-ram-v0')
    print(env.action_space)
    print(env.observation_space.high.shape)
    training_round = 100000
    training_length = 400
    '''
    observation = env.reset()

    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())

    '''
    
    dqn = ReinforcementModel(env)




    for i_episode in range(training_round):
        observation = env.reset().reshape(1, 128)
        for t in range(training_length):
            #print(observation[0][17], observation[0][18])
            action = dqn.act(observation)
            env.render()
            #            print(observation)
            
            newState, reward, done, info = env.step(action)
            #if done:
                #reward = -reward
            newState = newState.reshape(1, 128)
            #print(action)
            #if(newState[0][13] - observation[0][13] == 1):
            #    reward = -reward
            dqn.memAdd(observation, action, reward, newState, done)
            dqn.replay()
            observation = newState
            if done:
                break
        '''
        if t < 199:
            print(str(i_episode) + ' round, Failed to complete trial with step: ' + str(t))
        else:
            print(str(i_episode) + 'Success')
        '''

