# Reinforcement Learning Concepts

![Cartpole Example](https://cdn-images-1.medium.com/max/1600/1*oMSg2_mKguAGKy1C64UFlw.gif)


## What is Reinforcement Learning

Reinforcement learning is teaching a neural network to learn how to do specific tasks from an environment, without the need of any supplied data or information; it relies on the environment to teach it what it's supposed to do, and it will receive punishments/rewards for stuff that it is/isn't supposed to do. 


## What does the Neural Network look like in Reinforcement Learning?

Neural networks actually don't really change that much! Instead, it's simply the way neural networks are taught that changes. If you know how to make any of the previous neural networks in our tutorial (see the [main README](https://github.com/pdemange/Machine-Learning-Research)), then you can try reinforcement learning! In our example programs, we each use a simple classification neural network. 


## How does Reinforcement Learning work? 

![Q-Policy Equation](https://cdn-images-1.medium.com/max/800/1*IyrdggWA1wsce4nBA6u2VA.png)

Although the equation looks scary, it can be broken down pretty simply: When the network performs an action, it's given a reward "r". Then, for the given action, it looks ahead to the next state, and predicts the best action that will result in the biggest reward. After that, it will multiply the reward it finds for the next state by a learning decay variable (so the network won't get used to having the same reward for the action every time). Finally, we just add the computed future reward with the current reward received, and that new value becomes the new reward for the given state and action the network just took!

In code, it looks like this: 

```Python
#reward == current reward given for the most recent action
#gamma == .95, the "decay" of the future reward

 qUpdate = reward + max(self.network.predict(nextState)[0]) * self.gamma
 
#Then we have it predict the recent action again, off of the most recent state:

qPred = self.network.predict(state)

#Now we give it the new, better or worse reward for that state and action

qPred[0][action] = qUpdate

#Finally, teach it the new reward for that action and state

network.fit(state, qPred)
```

## Memory in Reinforcement Learning

