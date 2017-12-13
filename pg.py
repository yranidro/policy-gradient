import gym
import gym_gomoku
import numpy as np
import os.path
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, MaxPooling2D, Conv2D, BatchNormalization
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D
import tensorflow as tf


STATE_SIZE = (19, 19)


class PGAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size #vector from 2D matrix
        self.action_size = action_size #vector
        self.gamma = 1
        self.learning_rate = 1e-3
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.model = self._build_model()
        self.model.summary()

    def _build_model(self): #need to be fixed
        model = Sequential()
        # model.add(Reshape((1, STATE_SIZE[0], STATE_SIZE[1]), input_shape=(self.state_size,)))
        # model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode='same',
        #                         activation='relu', init='he_uniform', data_format="channels_first"))
        # model.add(MaxPooling2D((2,2)))
        # model.add(Convolution2D(32, 3, 3, subsample=(1, 1), border_mode='same',
        #                         activation='relu', init='he_uniform', data_format="channels_first"))
        # model.add(MaxPooling2D((2,2)))
        # model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same',
        #                         activation='relu', init='he_uniform', data_format="channels_first"))
        # model.add(MaxPooling2D((2,2)))
        
        model.add(Reshape((STATE_SIZE[0], STATE_SIZE[1], 1), input_shape=(self.state_size,)))
        model.add(BatchNormalization())
        model.add(Conv2D(16, (3, 3), dilation_rate=(1, 1), border_mode='same',
                                activation='relu', kernel_initializer='he_uniform'))
        model.add(MaxPooling2D((2,2)))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), dilation_rate=(1, 1), border_mode='same',
                                activation='relu', kernel_initializer='he_uniform'))
        model.add(MaxPooling2D((2,2)))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), dilation_rate=(1, 1), border_mode='same',
                                activation='relu', kernel_initializer='he_uniform'))
        model.add(Flatten())
        model.add(BatchNormalization())  
        # model.add(Dense(self.action_size, activation='relu', init='he_uniform'))
        model.add(Dense(self.action_size, activation='softmax'))
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model

    def remember(self, state, action, prob, reward):
        y = np.zeros([self.action_size])
        y[action] = 1/prob[action]
        #self.gradients.append(np.array(y).astype('float32') - prob) #??
        self.gradients.append(np.array(y).astype('float32'))
        self.states.append(state)
        self.rewards.append(reward)

    def act(self, state):
        state = state.reshape([1, state.shape[0]]) #??
        aprob = self.model.predict(state, batch_size=1).flatten()
        # print aprob, np.sum(aprob)
        
        validActions = env.action_space.valid_spaces
        bprob = np.copy(aprob)
        aprob[list(set(range(self.action_size))-set(validActions))] = 0
        prob = aprob / np.sum(aprob)
        #if np.sum(aprob) == 0 or np.isnan(np.sum(aprob)):
        
        action = np.random.choice(self.action_size, 1, p=prob)[0]

        # print aprob
        # print "\n lalal\n", bprob
        # print("action = {} prob = {} {}  max prob = {}".format(action, bprob[action], prob[action], np.amax(prob)))
        # print "prob sum = ", np.sum(aprob), "all prob sum = ", np.sum(bprob)
        # #print "valid = ", validActions
        # raw_input()

        self.probs.append(prob)
        return action, prob

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            # if rewards[t] != 0:
            #     running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def train(self):
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards)
        #rewards = rewards / np.std(rewards - np.mean(rewards))
        gradients *= rewards
        X = np.squeeze(np.vstack([self.states]))
        # Y = self.probs + self.learning_rate * np.squeeze(np.vstack([gradients])) #??
        Y = -self.learning_rate * np.squeeze(np.vstack([gradients]))
        self.model.train_on_batch(X, Y)
        self.states, self.probs, self.gradients, self.rewards = [], [], [], []

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def preprocess(I):
    return I.astype(np.float).ravel()

if __name__ == "__main__":
    # writer = tf.summary.FileWriter('board_beginner')  # create writer
    # t_score = tf.constant(0)
    # t_numSteps = tf.constant(0)
    # writer.add_summary(t_score, episode)
    # writer.add_summary(t_score, episode)
    
    env = gym.make('Gomoku19x19-v0') # default 'beginner' level opponent policy
    state = env.reset()
    score = 0
    episode = 0
    numSteps = 0
    winGames = 0

    state_size = STATE_SIZE[0] * STATE_SIZE[1]
    action_size = env.action_space.n
    agent = PGAgent(state_size, action_size)
    if os.path.exists('gomoku.h5'):
        agent.load('gomoku.h5')
    while True:
        #env.render()

        x = preprocess(state)
        action, prob = agent.act(x)
        state, reward, done, info = env.step(action)
        score += reward
        agent.remember(x, action, prob, reward)
        numSteps += 1

        if done:
            episode += 1
            agent.train()
            if score > 0: winGames += 1
            print('Episode: %d - Score: %f. - Steps: %d - Win: %d' % (episode, score, numSteps, winGames))

            score = 0
            numSteps = 0
            state = env.reset()
            if episode > 1 and episode % 10 == 0:
                agent.save('gomoku.h5')
