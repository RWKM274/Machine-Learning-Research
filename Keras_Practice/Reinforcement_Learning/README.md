# Reinforcement Learning Concepts: Deep Q-Policy Network (DQN)

![Cartpole Example](https://cdn-images-1.medium.com/max/1600/1*oMSg2_mKguAGKy1C64UFlw.gif)


## What is Reinforcement Learning

Reinforcement learning is teaching a neural network to learn how to do specific tasks from an environment, without the need of any supplied data or information; it relies on the environment to teach it what it's supposed to do, and it will receive punishments/rewards for stuff that it is/isn't supposed to do. 


## What does the Neural Network look like in Reinforcement Learning?

Neural networks actually don't really change that much! Instead, it's simply the way neural networks are taught that changes. If you know how to make any of the previous neural networks in our tutorial (see the [main README](https://github.com/pdemange/Machine-Learning-Research)), then you can try reinforcement learning! In our example programs, we each use a simple classification neural network. 


## How does Reinforcement Learning work? 

![Q-Policy Equation](https://cdn-images-1.medium.com/max/800/1*IyrdggWA1wsce4nBA6u2VA.png)

Although the equation looks scary, it can be broken down pretty simply: When the network performs an action, it's given a reward. Then, for the given action, it looks ahead to the next state, and predicts the best action that will result in the biggest reward. After that, it will multiply the reward it finds for the next state by a learning decay variable (so the network won't get used to having the same reward for the action every time). Finally, we just add the computed future reward with the current reward received, and that new value becomes the new reward for the given state and action the network just took!

In code, it looks like this: 

```Python
#reward == current reward given for the most recent action
#gamma == .95, the "decay" of the future reward
#network == The neural network model being trained 

 qUpdate = reward + max(network.predict(nextState)[0]) * self.gamma
 
#Then we have it predict the recent action again, off of the most recent state:

qPred = network.predict(state)

#Now we give it the new, better or worse reward for that state and action

qPred[0][action] = qUpdate

#Finally, teach it the new reward for that action and state

network.fit(state, qPred)
```

## Memory in Reinforcement Learning

Reinforcement networks tend to be "forgetful", as in, they don't remember the moves they made in the past to lead them to a victory or a failure! So, we need to give the network a "memory" where it can "remember" a number of it's past moves, that way it doesn't forget what it knows. To do this, a deque is implemented that will keep track of every move it has made recently, with the state it was in, the action it took, the reward it got, the next state it was in, and whether or not it completed the game or lost. Of course, these parameters will be different depending on what they network will be doing, but within our example programs this is how we defined them. 

To implement this "memory" in code, it would look like this:

```Python

#Make a deque, with a max "memory" length of 2000 (most recent 2000 moves)

memory = deque(maxlen=2000)

#Now, a function is made to just "remember" (or add) any moves to the memory deque

#state == previous state
#action == the action it took for the previous state
#reward == reward given for the action in that state
#newState == the new state it's in
#done == whether or not the game is done (in the example programs that we made, it loses if this is the case)

memAdd(state, action, reward, newState, done):
    self.memory.append([state, action, reward, newState, done])
    
#Now, to train it off of it's memories, we'll have it sample 32 moves from the memory.

samples = random.sample(memory, 32)

#Now, we teach it for those moves

for sample in samples: 
    state, action, reward, nextState, done = sample
    qUpdate = reward + max(network.predict(nextState)[0]) * self.gamma
    qPred = network.predict(state)
    qPred[0][action] = qUpdate
    network.fit(state, qPred)
    
```


