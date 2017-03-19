import random, numpy, math, gym
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import keras
import matplotlib.pyplot as plt
from SumTree import SumTree

# hyper params
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64
UPDATE_TARGET_FREQUENCY = 5000

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
        self.model_target = self._createModel()
        if loadModel:
            self.model.load_weights(modelFilename)

    def updateTargetModel(self):
        '''
        Update target network
        '''
        self.model_target.set_weights(self.model.get_weights())

    def _createModel(self):
        model = Sequential()

        model.add(Dense(64, activation='relu', input_dim=self.stateSize))
        model.add(Dense(self.actionSize, activation='linear'))

        opt = RMSprop(lr=0.00025)
        model.compile(loss='mse', optimizer=opt)

        return model

    def train(self, x, y, epoch=1, verbose=0):
        self.model.fit(x, y, batch_size=64, nb_epoch=epoch, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.model_target.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s):
        return self.predict(s.reshape(1, self.stateSize)).flatten()

class Memory:
    e = 0.01
    a = 0.6

    samples = []
    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample): # in (s,a,r,s)
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []

        segment = self.tree.total() / n

        for i in range(n):
            a = segment * 1
            b = segment * (i + 1)

            s = random.uniform(a,b)
            (idx, p, data) = self.tree.get(s)
            batch.append( (idx, data) )


        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)

class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateSize, actionSize):
        self.stateSize = stateSize
        self.actionSize = actionSize

        self.network = Network(stateSize, actionSize)

    def act(self, s, testMode = False):
        if random.random() < self.epsilon and not testMode:
            return random.randint(0, self.actionSize-1)
        else:
            return numpy.argmax(self.network.predictOne(s))

    def observe(self, sample):
        x, y, errors = self._get_td_target([(0, sample)])
        self.memory.add(errors[0], sample)


        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.network.updateTargetModel()

        self.steps +=1
        self.epsilon  = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def _get_td_target(self, batch):
        batchLen = len(batch)
        null_state = numpy.zeros(self.stateSize)

        states = numpy.array([ o[1][0] for o in batch])
        states_ = numpy.array([ (null_state if o[1][3] is None else o[1][3] ) for o in batch])

        p = self.network.predict(states)
        p_ = self.network.predict(states_, target= False)
        pTarget_ = self.network.predict(states_, target= True)

        x = numpy.zeros((batchLen, self.stateSize))
        y = numpy.zeros((batchLen, self.actionSize))
        errors = numpy.zeros(batchLen)

        for i in range(batchLen):
            o = batch[i][1]
            s = o[0]; a = o[1]; r = o[2]; s_= o[3]

            # copy old prob. of each actions
            t = p[i] # t is the temp holder
            oldActionValue = p[i][a]
            if s_ is None:
                t[a] = r
            else:
                # bootstraping
                t[a] = r + GAMMA * pTarget_[i][numpy.argmax(p_[i])]

            x[i] = s
            y[i] = t
            errors[i] = abs(oldActionValue - t[a])
        return (x, y, errors)

    def replay(self):
        batch = self.memory.sample(BATCH_SIZE) # in  (idx )
        x, y, errors = self._get_td_target(batch)

        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])

        self.network.train(x,y)

class RandomAgent:
    memory = Memory(MEMORY_CAPACITY)
    steps = 0
    def __init__(self, actionSize):
        self.actionSize = actionSize
        self.steps = 0
    def act(self, s, testMode = False):
        return random.randint(0,self.actionSize-1)
    def observe(self, sample):  # in s a r s_
        error = abs(sample[2])
        self.memory.add(error, sample)
        self.steps+=1
    def replay(self):
        pass

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
randomAgent = RandomAgent(actionSize)

try:
    print("ini exp ")
    while randomAgent.steps < MEMORY_CAPACITY:
        env.run(randomAgent)
        
        print("randomAgent fill "+str( randomAgent.steps)+ "/", MEMORY_CAPACITY)

    agent.memory = randomAgent.memory
    randomAgent = None

    print("ready for learning")

    for i in range(2000):
        env.run(agent)
        if i % 10 == 0 :
            env.run(agent, True)
    env.plotRewards()
finally:
    agent.network.model.save("cartpole-DDQN-P-REPLAY.h5")
