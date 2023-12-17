# DDQN_Mario

## Introduction / 介绍
This the RL course project

## Features
- Implement the thompson sampling/Greedy/Softmax as the exploration startegy
- Play with the hyperparameters
- Introduce different Neural Networks


## Installation
Just install the environment.yml
Cuda toolkit is 11.3 version

## Run
Simply run the ipynb file is fine

## Code part
For following two segments of code, imply copy and paste to replace the [def act] in original notebook Mario class part is fine.  
Then, you can use these two exploration strategy.


#### Softmax
```python
import torch.nn.functional as F

    def act(self, state):
        # Convert state to tensor and move to device
        state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
        state = torch.tensor(state, device=self.device).unsqueeze(0)

        # Get action values from the network
        action_values = self.net(state, model="online")

        # Apply softmax to get action probabilities
        action_probs = F.softmax(action_values, dim=-1).cpu().data.numpy().squeeze()

        # Select an action based on the probability distribution
        action_idx = np.random.choice(np.arange(self.action_dim), p=action_probs)

        # Increment step
        self.curr_step += 1
        return action_idx
```
#### Greedy
```python
    def act(self, state):
        """
    Given a state, choose an epsilon-greedy action and update value of step.

    Inputs:
    state(``LazyFrame``): A single observation of the current state, dimension is (state_dim)
    Outputs:
    ``action_idx`` (``int``): An integer representing which action Mario will perform
    """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx
```
#### The hyperparameters can be adjusted directly
#### We discard the vgg16 code in this repository since it simply increase the running time but not increase the performance of the algorithm.
