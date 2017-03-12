# -*- coding: utf-8 -*-
import gym
env = gym.make('CartPole-v0')
env.reset()
env.render()
import os
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import math


memory_capacity = 100000
nbReplay = 30
td_discount_rate = 0.99
learningRate = 0.0001
epsilonGreedy = 0.001
epsilonGreedyDecayRate = 0.000001


num_episodes = 50000
logDir = "logs/" + time.strftime("%Y%m%d%H%M%S", time.localtime())
memoryCapacity = 100000


class Network():
    def __init__(self, stateSize, actionSize, learningRate= 0.0001):
        self.stateSize = stateSize
        self.learningRate = learningRate
        self.actionSize = actionSize
        self._createModel()
        self.steps = 0
        
    def _createModel(self):
        self.input = tf.placeholder('float', shape=[None,self.stateSize])      
        x1 = slim.fully_connected(self.input, 64, scope='fc/fc_1')
        x1 = tf.nn.relu(x1)
        self.Qout = slim.fully_connected(x1, self.actionSize)
        
        self.tdTarget = tf.placeholder(shape=[None],dtype=tf.float32)
        self.loss = tf.reduce_mean(tf.square(self.tdTarget - tf.reduce_max(self.Qout, axis=1) ) )   
        
#        self.trainer = tf.train.AdamOptimizer(learning_rate=self.learningRate)
        self.trainer = tf.train.RMSPropOptimizer(learning_rate=self.learningRate)
        self.updateModel = self.trainer.minimize(self.loss)
        
        
        tdTargetLogger= tf.summary.scalar('tdTarget', tf.reduce_mean(self.tdTarget))
        lossLogger= tf.summary.scalar('loss', self.loss)
        self.log = tf.merge_summary([tdTargetLogger, lossLogger])
    
    def train(self, sess, states, tdTargets, logWriter):              
        graphRun = [self.updateModel, self.log]
        _, summary = sess.run(graphRun,
                            feed_dict={self.input: states,
                                       self.tdTarget : tdTargets})
        
        logWriter.add_summary(summary, self.steps)
        self.steps += 1
    
    def predict(self, sess, states):
        return sess.run(self.Qout, feed_dict={self.input: states})
        
    def predictOne(self, sess, state):
        return sess.run(self.Qout, feed_dict={self.input: [state]}) [0]


class Memory():
    buffer = [] 

    def __init__(self, capacity):
        self.capacity = capacity
        
    def add(self, sample):
        '''     in [ s, a, r, s_] fromat  '''
        self.buffer.append(sample)
        
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
            
    def sample(self, n):
        n = min(n, len(self.buffer))
        return random.sample(self.buffer, n)
        
class Agent():
    def __init__(self, 
                 stateSize,
                 actionsSize, 
                 td_discount_rate = 0.99, 
                 learningRate= 0.0001,
                 memoryCapacity = 100000,
                 min_epsilon = 0.001,
                 epsilon_decay = 0.00001,
                 batchSize = 64,
                 max_epsilon = 1):                #epsilonGreedy
    
        self.stateSize = stateSize
        self.actionsSize = actionsSize
        self.td_discount_rate = td_discount_rate
        self.learningRate = learningRate
        self.epsilon = max_epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.memoryCapacity = memoryCapacity
        self.batchSize = batchSize
        
        self.network = Network(stateSize, actionsSize, learningRate)
        self.memory = Memory(memoryCapacity)
        
        
        self.epsilonTensor = tf.placeholder(dtype=tf.float32)
        self.epsilonLogger= tf.summary.scalar('epsilon', self.epsilonTensor)
    
    def act(self, sess, logWriter, state):
        _= sess.run(self.epsilonLogger, feed_dict={self.epsilonTensor: self.epsilon})
        logWriter.add_summary(_, self.network.steps)
        
        if random.random() < self.epsilon:
            return random.randint(0, self.actionsSize-1)
        else:
            return np.argmax(self.network.predictOne(sess, state))
            
    def addMemory(self, state, action, reward, newState):
        if newState == None:
            reward = -100
        self.memory.add([state, action, reward, newState])
    
    def reduceEpsilon(self, episode):        
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * math.exp(-self.epsilon_decay * episode)
        
        
    def replay(self, sess, logWriter):
        batch = self.memory.sample(self.batchSize)
        batchSize = len(batch)
        
        nullState = np.zeros(self.stateSize)
        
#        states = batch[:,0]
        states_ = np.array([ (nullState if o[3] is None else o[3]) for o in batch ])
        
#        probs = self.network.predict(states)
        probs_ = self.network.predict(sess, states_)
        
        inputStates = np.zeros((batchSize, self.stateSize))
        inputTdTargets = np.zeros((batchSize))
        
        for i in range(batchSize):
            row = batch[i]
            s = row[0]; r = row[2]; s_ = row[3]; # a = row[1]; 
            
            tdTarget = r
            if s_ is not None:
                # bootstrap target TD(0)
                tdTarget += self.td_discount_rate * np.amax(probs_[i]) 
            
            inputStates[i] = s
            inputTdTargets[i]  = tdTarget

        self.network.train(sess, inputStates, inputTdTargets, logWriter)

            
        
print("state size is " + str(env.observation_space.shape[0]))
print("action size is " + str(env.action_space.n))
print("----------------------------------------------------")
        
agent = Agent(env.observation_space.shape[0], env.action_space.n, memoryCapacity=memoryCapacity, learningRate=0.1)

episodeRewardTensor = tf.placeholder(tf.float32, name="episodeReward")
episodeRewardSummary = tf.summary.scalar('episodeReward', episodeRewardTensor)

init = tf.global_variables_initializer()


with tf.Session() as sess:
    writer = tf.summary.FileWriter(logDir, sess.graph)
    sess.run(init)
    
    for episode in range(num_episodes):      
        done = False
        observation = env.reset()
        oldObservation = observation
        
        episodeReward = 0
        while not done:    
            if episode % 100 == 0 :
                env.render()    
            action = agent.act(sess, writer, observation)
            s_, reward, done, info = env.step(action)
            agent.addMemory(oldObservation, action, reward, None if done else s_)
            agent.replay(sess, writer)
            
            oldObservation = s_
            episodeReward += reward
        
        agent.reduceEpsilon(episode)
        episodeRewardSummary_ = sess.run(episodeRewardSummary, feed_dict={episodeRewardTensor: episodeReward })
        writer.add_summary(episodeRewardSummary_, episode)
            
        if episode%100 == 0:
            print("episode " + str(episode) +" stpes "+ str(agent.network.steps) +" .with total reward "+str(episodeReward)     )
    print("training end ")        
    env.close()