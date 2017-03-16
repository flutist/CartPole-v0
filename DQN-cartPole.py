import random, numpy, math, gym
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import keras
import matplotlib.pyplot as plt

# hyper params
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001

PROBLEM = 'CartPole-v0'


class Network:
    def __init__(self, stateSize, actionSize, loadModel = False, modelFilename="cartpole-basic.h5" ):
        self.stateSize = stateSize
        self.actionSize = actionSize

        self.model = self._createModel()
        if loadModel:
            self.model.load_weights(modelFilename)

    def _createModel(self):
        model = Sequential()

        model.add(Dense(64, activation='relu', input_dim=self.stateSize))
        model.add(Dense(self.actionSize, activation='linear'))

        opt = RMSprop(lr=0.00025)
        model.compile(loss='mse', optimizer=opt)

        return model

    def train(self, x, y, epoch=1, verbose=0):
        self.model.fit(x, y, batch_size=64, nb_epoch=epoch, verbose=verbose)

    def predict(self, s):
        return self.model.predict(s)

    def predictOne(self, s):
        return self.predict(s.reshape(1, self.stateSize)).flatten()

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
    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem)
        self.rewardHistory = []

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
        print("Total reward:", R)

    def plotRewards(self):
        plt.figure()
        plt.plot(self.rewardHistory)

env = Environment(PROBLEM)

stateSize = env.env.observation_space.shape[0]
actionSize = env.env.action_space.n
agent = Agent(stateSize, actionSize)

try:
    for i in range(1500):
        env.run(agent)
        if i % 10 == 0 :
            env.run(agent, True)
    env.plotRewards()
finally:
    agent.network.model.save("cartpole-basic.h5")
