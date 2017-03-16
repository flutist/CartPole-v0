# this is the version of tensoflow
import random, numpy, math, gym
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time

# hyper params
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001

PROBLEM = 'CartPole-v0'
logDir = "logs/" + time.strftime("%Y%m%d%H%M%S", time.localtime())
totalSteps = 0

class Network:
    sess = None
    logWriter = None
    def __init__(self, stateSize, actionSize):
        self.stateSize = stateSize
        self.actionSize = actionSize

        self._createModel()

    def _createModel(self):

        self.input = tf.placeholder('float', shape=[None,self.stateSize])
        x1 = slim.fully_connected(self.input, 64, scope='fc/fc_1')
        x1 = tf.nn.relu(x1)
        self.Qout = slim.fully_connected(x1, self.actionSize)

        self.tdTarget = tf.placeholder(shape=[None, self.actionSize],dtype=tf.float32)
        self.loss = tf.reduce_mean(tf.square(self.tdTarget - self.Qout ) )

        self.trainer = tf.train.RMSPropOptimizer(learning_rate=0.00025)
        self.updateModel = self.trainer.minimize(self.loss)


        tdTargetLogger= tf.summary.scalar('tdTarget', tf.reduce_mean(self.tdTarget))
        lossLogger= tf.summary.scalar('loss', self.loss)
        self.log = tf.summary.merge([tdTargetLogger, lossLogger])

    def train(self, x, y):
        graphRun = [self.updateModel, self.log]
        _, summary = self.sess.run(graphRun,
                            feed_dict={self.input: x,
                                       self.tdTarget : y})
        self.logWriter.add_summary(summary, totalSteps)

    def predict(self, s):
        return self.sess.run(self.Qout, feed_dict={self.input: s})

    def predictOne(self, s):
        return self.sess.run(self.Qout, feed_dict={self.input: [s]}) [0]

class Memory:
    samples = []
    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample): # in (s,a,r,s)
        self.samples.append(sample)

        if (len(self.samples)> self.capacity):
            self.samples.pop(0)

    def sample(self, n):
        n = min (n, len(self.samples))
        return random.sample(self.samples, n)

class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateSize, actionSize):
        self.stateSize = stateSize
        self.actionSize = actionSize

        self.network = Network(stateSize, actionSize)
        self.memory = Memory(MEMORY_CAPACITY)

    def act(self, s, testMode = False):
        if random.random() < self.epsilon and not testMode:
            return random.randint(0, self.actionSize-1)
        else:
            return numpy.argmax(self.network.predictOne(s))

    def observe(self, sample):
        self.memory.add(sample)

        self.steps +=1
        totalSteps = self.steps
        self.epsilon  = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def replay(self):
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        null_state = numpy.zeros(self.stateSize)

        states = numpy.array([ o[0] for o in batch])
        states_ = numpy.array([ (null_state if o[3] is None else o[3] ) for o in batch])

        p = self.network.predict(states)
        p_ = self.network.predict(states_)

        x = numpy.zeros((batchLen, self.stateSize))
        y = numpy.zeros((batchLen, self.actionSize))

        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_= o[3]

            # copy old prob. of each actions
            t = p[i] # t is the temp holder
            if s_ is None:
                t[a] = r
            else:
                # bootstraping
                t[a] = r + GAMMA * numpy.amax(p_[i])

            x[i] = s
            y[i] = t
        self.network.train(x,y)

class Environment:
    sess = None

    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem)
        self.rewardHistory = []
        self.episodeRewardTensor = tf.placeholder(tf.float32, name="episodeReward")
        self.episodeRewardSummary = tf.summary.scalar('episodeReward', self.episodeRewardTensor)

    def run(self, agent, testMode= False):
        s = self.env.reset()
        R = 0

        while True:
            a = agent.act(s, testMode)

            s_, r, done, info = self.env.step(a)

            if done:
                s_ = None

            agent.observe((s, a, r, s_))
            agent.replay()

            s = s_
            R += r

            if done:
                break
        if testMode:
            self.rewardHistory.append(R)
        else:
            print("Total reward:", R)
            return self.sess.run(self.episodeRewardSummary, feed_dict={self.episodeRewardTensor: R })

    def plotRewards(self):
        plt.figure()
        plt.plot(self.rewardHistory)




env = Environment(PROBLEM)

stateSize = env.env.observation_space.shape[0]
actionSize = env.env.action_space.n
agent = Agent(stateSize, actionSize)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    writer = tf.summary.FileWriter(logDir, sess.graph)
    sess.run(init)

    agent.network.sess = sess
    env.sess = sess
    agent.network.logWriter = writer

    for i in range(3000):
        episodeRewardSummary_ = env.run(agent)
        if i % 10 == 0 :
            env.run(agent, True)
        writer.add_summary(episodeRewardSummary_, i)
    env.plotRewards()
    print("-------------done-------------")
