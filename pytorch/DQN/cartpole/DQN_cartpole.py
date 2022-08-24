#!/usr/bin/env python
import rospy
import gym
import random
import numpy as np
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

## Memory Replay
Transition = namedtuple('Transition', ('state','next_state','action','reward','mask'))

class Memory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, next_state, action, reward, mask):
        self.memory.append(Transition(state, next_state, action, reward, mask))
        
    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        return batch
    
    def __len__(self):
        return len(self.memory)

## Model
class DQN(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super(DQN, self).__init__()
        self.net = nn.Sequential(                
            nn.Linear(num_inputs, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_outputs), 
            )

    def forward(self, x):
        return self.net(x)

class DQN_Solver():
    def __init__(self, monitor=False):
                 
                self.env = gym.make('CartPole-v1') #openAI-Gym
                self.cum_rewards = []
                self.cum_running_score = []
                self.epsilon_cum = []
                self.avg_loss = []
                self.losses = []
                self.duration = 0

    ## E-greedy policy
    def select_action(self, epsilon, state, net):
        if np.random.random() < epsilon:
            ran_action = self.env.action_space.sample()
            return ran_action
        else:
            _ , max_action = net(state).max(-1)
            return max_action.item()

    ## Target update
    def update_target_model(self, policy_net, target_net):
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

    ## Target 
    def smooth_update_target_model(self, policy_net, target_net, update_target):
        rho = 1 - (1/update_target) #decay rate 
        for p_tgt, p in zip(target_net.parameters(),policy_net.parameters()):
            p_tgt.data.mul_(1.0 - rho)
            p_tgt.data.add_(rho*p.data) 

    ## Train model
    def train_model(self, policy_net, target_net, batch, optimizer,loss_fn, gamma):
        states = torch.stack(batch.state)
        next_states = torch.stack(batch.next_state)
        actions = torch.stack(batch.action).squeeze(1).float() 
        rewards = torch.tensor(batch.reward)
        masks = torch.tensor(batch.mask)
        
        qvalues = policy_net(states).squeeze(1)
        next_qvalues = target_net(next_states).squeeze(1)
        q_est = qvalues.mul(actions).sum(1)
        
        with torch.no_grad():
            target = rewards + masks*gamma*next_qvalues.max(1)[0]
            
        loss = loss_fn(q_est, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()

    def main(self):
        start_time = time.time()
        
        ## Configurations
        #hyperameters
        batch_size = 64 
        update_target = 200 
        replay_memory_capacity = 1000000 
        memory = Memory(replay_memory_capacity)
        initial_exploration = 2000 
        gamma = 0.995
        lr = 1.0e-3 
        epsilon = 1.0
        num_episodes =  500
        steps = 0
        print_every = 50

        #model 
        num_inputs = 4
        num_hidden = 256  
        num_outputs = 2

        #declare model
        policy_net = DQN(num_inputs, num_hidden, num_outputs) 
        target_net = DQN(num_inputs, num_hidden, num_outputs) 
        self.update_target_model(policy_net, target_net) #copy weights

        #optimizer & loss function
        optimizer = optim.Adam(policy_net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()


        for episode in range(num_episodes):
            done = False
            
            score = 0
            obs = self.env.reset()
            state = torch.tensor(obs).unsqueeze(0)
            
            while not done:
                steps += 1
                action = self.select_action(epsilon, state, policy_net)
                next_obs, reward, done, _ = self.env.step(action)
                next_state = torch.tensor(next_obs).unsqueeze(0)
                
                mask = 0 if done else 1
                reward = -reward if done else reward
                score += reward
                
                action_one_hot = np.zeros(2)
                action_one_hot[action] = 1
                action_one_hot = torch.tensor(action_one_hot).unsqueeze(0)
                
                memory.push(state, next_state, action_one_hot, reward, mask)
                state = next_state
                
                if steps > initial_exploration:
                    epsilon = max(epsilon*0.9995, 0.01)
                    batch = memory.sample(batch_size)
                    loss = self.train_model(policy_net, target_net, 
                                    batch, optimizer, loss_fn, gamma)
                    self.losses.append(loss)
                    
                #smooth_tgt_update
                self.smooth_update_target_model(policy_net, 
                                        target_net, update_target)
                            
            if self.losses: #execute if not empty
                self.avg_loss.append( np.mean(self.losses) )
            
            self.epsilon_cum.append(epsilon)
            
            score = score #if score == 500.0 else score #+ 1
            self.cum_rewards.append(score)
            avg_reward = np.mean(self.cum_rewards[-10:])
            self.cum_running_score.append(avg_reward)
            
            if episode % print_every == 0:
                print("Episode: {} | Avg_reward: {}".format(episode,avg_reward))
            
        end_time = time.time()
        self.duration = end_time - start_time
        #self.env.close() #close simulator

    def plotting(self):
        ## Plots
        plt.figure(figsize=[15,8])
        plt.subplot(3,2,1)
        plt.plot(self.cum_rewards)
        plt.ylabel('rewards')
        #plt.xlabel('episodes')

        plt.subplot(3,2,2)
        plt.plot(self.cum_running_score)
        plt.ylabel('rolling_reward')
        #plt.xlabel('episodes')

        plt.subplot(3,2,3)
        plt.plot(self.epsilon_cum) 
        plt.ylabel('epsilon')
        plt.xlabel('episodes')

        plt.subplot(3,2,4)
        plt.plot(self.avg_loss) 
        plt.ylabel('avg_losses')
        plt.xlabel('episodes')

        # save the figure
        plt.savefig('plots.png', dpi=300, bbox_inches='tight')
        #plt.show()

    def keep_time(self):
        if self.duration > 3600: 
            #print('{:.2f} hrs'.format(duration/3600))
            line = 'Script run for {:.2f} hrs'.format(self.duration/3600)
        else:
            #print('{:.2f} min'.format(duration/60))
            line = 'Script run for {:.2f} min'.format(self.duration/60)

        #write to file
        outF = open('details.txt', 'w')
        outF.write(line)
        outF.close()

if __name__ == '__main__':
    agent = DQN_Solver()
    agent.main()
    agent.plotting()
    agent.keep_time()
