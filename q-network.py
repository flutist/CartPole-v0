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



num_episodes = 100000
memory_capacity = 100000
nbReplay = 30
td_discount_rate = 0.99
learningRate = 0.0001
epsilonGreedy = 0.1
epsilonGreedyDecayRate = 0.00001
logDir = "logs/" + time.strftime("%Y%m%d%H%M%S", time.localtime())


class Memory():
    buffer = [] 

    def __init__(self, capacity):
        self.capacity = capacity
    def add(self, sample):
        '''
            in [ x, a, r, s_] fromat
        '''
        self.buffer.append(sample)
        
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
            
    
    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.buffer, n)

class Agent():
    def __init__(self, actions, td_discount_rate = 0.99, learningRate= 0.0001, epsilonGreedy = 0.1):
        self.learningRate = learningRate
        self.td_discount_rate = td_discount_rate
        self.epsilonGreedy = epsilonGreedy
        
        self.input = tf.placeholder('float', shape=[None,4])      
        x1 = slim.fully_connected(self.input, 32, scope='fc/fc_1')
        x1 = tf.nn.relu(x1)
        self.Qout = slim.fully_connected(x1, actions)
        
        self.predict = tf.argmax(self.Qout,1)
        self.logQVal = tf.summary.scalar('QVal', tf.reduce_mean(self.predict) )
        
        # get the best action q values 
        self.newQout = tf.placeholder(shape=[None,2],dtype=tf.float32)
        self.epsilonInput = tf.placeholder(dtype=tf.float32, name="epsilonInput")
        self.newstateReward = tf.placeholder(shape=[None],dtype=tf.float32)
        self.tdTarget = self.newstateReward + td_discount_rate * np.amax(self.newQout)
        self.td_error = tf.square(self.tdTarget - np.amax(self.Qout))
        # trun into single scalar value 
        self.loss = tf.reduce_mean(self.td_error)        
        
        self.tdLogger= tf.summary.scalar('tdLoss', self.loss)
        self.tdTargetLogger= tf.summary.histogram('tdTarget', self.tdTarget)
        self.epsilonLogger= tf.summary.scalar('epsilon', self.epsilonInput)
        
        # minimize the loess (mean of td errors)
        self.trainer = tf.train.AdamOptimizer(learning_rate=self.learningRate)
        self.updateModel = self.trainer.minimize(self.loss)
        
        
        self.memory = Memory(memory_capacity)
    
    def predictAction(self, state, env):
        a,allQ, logQ = sess.run(
                          [self.predict, self.Qout, self.logQVal],
                          feed_dict={ 
                                     self.input:[state] 
                            }
                        )
        
        if np.random.rand(1) < self.epsilonGreedy:
            a =  [env.action_space.sample()]
        if self.epsilonGreedy > 0:
            self.epsilonGreedy -= epsilonGreedyDecayRate
        return a, allQ, logQ

    def reply(self):        
        batch = self.memory.sample(nbReplay)
        
        states = np.array([ o[0] for o in batch ])
        states_ = np.array([ (nbReplay if o[3] is None else o[3]) for o in batch ])
        
        p = agent.brain.predict(states)
        p_ = agent.brain.predict(states_)

        x = np.zeros((nbReplay, self.stateCnt))
        y = np.zeros((nbReplay, self.actionCnt))     
        
        for i in range(nbReplay):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + td_discount_rate * numpy.amax(p_[i])

            x[i] = s
            y[i] = t

        
        
    def train(self, sess, oldState, newStateQ, newstateReward):                
        graphRun = [self.updateModel, self.td_error, self.tdLogger, self.epsilonLogger, self.tdTargetLogger]
        return sess.run(graphRun,
                        feed_dict={self.input: [oldState],
                                   self.newQout : newStateQ,
                                   self.epsilonInput: self.epsilonGreedy,
                                   self.newstateReward: [newstateReward]
                                   })
        
            

agent = Agent(2)

#episodeError = tf.placeholder(tf.float32, name="episodeError")
#tf.summary.scalar('episodeError', episodeError)
episodeRewardTensor = tf.placeholder(tf.float32, name="episodeReward")
episodeRewardSummary = tf.summary.scalar('episodeReward', episodeRewardTensor)



init = tf.global_variables_initializer()


with tf.Session() as sess:
    writer = tf.summary.FileWriter(logDir, sess.graph)
    totalStep = 0
    sess.run(init)
    for episode in range(num_episodes):      
        done = False
        observation = env.reset()
        oldObservation = observation
        
        episodeReward = 0
        while not done:    
            env.render()
            action, qValues, qSummery =agent.predictAction(observation, env)            
            writer.add_summary(qSummery, totalStep)
            new_observation, reward, done, info = env.step(action[0])
            
            _, tdError,tdSummery, epsilonSummery,tdTargetSummery = agent.train(sess, oldObservation, qValues, reward)
            
            oldObservation = new_observation
            episodeReward += reward
            totalStep+=1
            
            writer.add_summary(tdSummery, totalStep)
            writer.add_summary(epsilonSummery, totalStep)
            writer.add_summary(tdTargetSummery, totalStep)
        
        episodeRewardSummary_ = sess.run(episodeRewardSummary,
                    feed_dict={episodeRewardTensor: episodeReward })
        writer.add_summary(episodeRewardSummary_, totalStep)
            
        if episode%100 == 0:
            print("episode " + str(episode) + " .with total reward "+str(episodeReward)     )
    print("training end ")        
    env.close()