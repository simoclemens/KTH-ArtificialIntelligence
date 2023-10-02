#!/usr/bin/env python3

from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import random
import math
import numpy as np


class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def __init__(self):
        super().__init__()
        self.revealed_fish = None
        self.probabilities = None
        self.observations = None
        self.models = None
        self.revealed_list = None

    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """

        # init one model for each fish type
        self.models = [HMMModel() for _ in range(7)]

        # create the observations matrix, with one line for each type of fish
        self.observations = [[] for _ in range(70)]

        # create probabilities matrix, with one row per type and one column per fish
        self.probabilities = [[0 for _ in range(70)] for _ in range(7)]

        # create a list of seven lists, each one containing the fishes of the given type
        self.revealed_fish = [[] for _ in range(7)]

        # create a simple list containing the indexes of fishes which have been found already
        self.revealed_list = []

    def insert_obs(self, observations):
        for i, obs in enumerate(observations):
            self.observations[i].append(obs)

    def argmax_matrix(self):
        max_prob = -1
        coord = (0, 0)

        for i, seq in enumerate(self.probabilities):
            for j, prob in enumerate(seq):
                if j not in self.revealed_list:
                    if prob > max_prob:
                        max_prob = prob
                        coord = (i, j)

        return coord

    def guess(self, step, observations):

        """
        This method gets called on every iteration, providing observations.
        Here the player should process and store this information,
        and optionally make a guess by returning a tuple containing the fish index and the guess.
        :param step: iteration number
        :param observations: a list of N_FISH observations, encoded as integers
        :return: None or a tuple (fish_id, fish_type)
        """

        self.insert_obs(observations)

        for i, obs in enumerate(self.observations):
            for j, m in enumerate(self.models):
                self.probabilities[j][i] = m.compute_seq_prob(obs)

        fish_type, fish_id = self.argmax_matrix()

        return fish_id, fish_type

    def reveal(self, correct, fish_id, true_type):
        """
        This methods gets called whenever a guess was made.
        It informs the player about the guess result
        and reveals the correct type of that fish.
        :param correct: tells if the guess was correct
        :param fish_id: fish's index
        :param true_type: the correct type of the fish
        :return:
        """

        self.revealed_fish[true_type].append(fish_id)
        self.revealed_list.append(fish_id)

        if len(self.observations[0]) > 2:
            for i, m in enumerate(self.models):
                for index in self.revealed_fish[i]:
                    m.train(self.observations[index])


def normalize(matrix, axis=1):
    matrix /= matrix.sum(axis=axis, keepdims=True)
    return matrix


class HMMModel:
    def __init__(self, hidden_states=8, n_obs=8):
        self.A = normalize(np.random.beta(a=1, b=1, size=(hidden_states, hidden_states))).tolist()
        self.B = normalize(np.random.beta(a=1, b=1, size=(hidden_states, n_obs))).tolist()
        self.p = normalize(np.random.beta(a=1, b=1, size=hidden_states), axis=0).tolist()

    def train(self, obs):

        # class -> alg
        A = self.A
        B = self.B
        p0 = self.p

        T = len(obs)
        N = len(A)
        K = len(B[0])
        MAX_ITER = 20

        mle = float("-inf")
        mle_new = float("-inf")
        iteration = 0

        while (mle_new > mle or mle_new == float("-inf")) and iteration <= MAX_ITER:
            mle = mle_new
            iteration += 1

            # ALPHA
            # initialization
            alpha = [[0 for _ in range(N)] for _ in range(T)]
            c_log = []
            c = []

            # first row
            for i in range(N):
                alpha[0][i] = p0[i] * B[i][obs[0]]

            c_t = 1 / sum(alpha[0])

            for i in range(N):
                alpha[0][i] *= c_t

            c.append(c_t)
            c_log.append(math.log(c_t))

            # computation
            for t in range(1, T):
                for i in range(N):
                    for j in range(N):
                        alpha[t][i] += alpha[t - 1][j] * A[j][i]
                    alpha[t][i] *= B[i][obs[t]]

                c_t = 1 / sum(alpha[t])

                for i in range(N):
                    alpha[t][i] *= c_t
                c.append(c_t)
                c_log.append(math.log(c_t))

            mle_new = -sum(c_log)

            # BETA
            # initialization
            beta = [[0 for _ in range(N)] for _ in range(T)]

            # first row
            for i in range(N):
                beta[T - 1][i] = c[T - 1]

            # computation
            for t in range(T - 2, -1, -1):
                for i in range(N):
                    for j in range(N):
                        beta[t][i] += beta[t + 1][j] * B[j][obs[t + 1]] * A[i][j]
                    beta[t][i] *= c[t]

            # DI-GAMMA/GAMMA

            # initialization
            di_gamma = [[[0 for _ in range(N)] for _ in range(N)] for _ in range(T)]
            gamma = [[0 for _ in range(N)] for _ in range(T)]

            # computation
            for t in range(T - 1):
                for i in range(N):
                    for j in range(N):
                        di_gamma[t][i][j] = alpha[t][i] * A[i][j] * B[j][obs[t + 1]] * beta[t + 1][j]
                    gamma[t][i] = 0
                    for j in range(N):
                        gamma[t][i] += di_gamma[t][i][j]

            # last row
            gamma[T - 1] = alpha[T - 1]

            # initial state update
            p0 = gamma[0]

            # A update
            for i in range(N):
                den = 0
                for t in range(T - 1):
                    den += gamma[t][i]
                for j in range(N):
                    num = 0
                    for t in range(T - 1):
                        num += di_gamma[t][i][j]
                    A[i][j] = num / den

            # B update
            for i in range(N):
                for k in range(K):
                    den = 0
                    num = 0
                    for t in range(T):
                        den += gamma[t][i]
                        if obs[t] == k:
                            num += gamma[t][i]
                    B[i][k] = num / den

        # alg -> class
        self.A = A
        self.B = B
        self.p = p0

    def compute_seq_prob(self, obs):
        # class -> alg
        A = self.A
        B = self.B
        p0 = self.p

        T = len(obs)
        N = len(A)

        alpha = [[0 for _ in range(N)] for _ in range(T)]

        for i in range(N):
            alpha[0][i] = p0[i] * B[i][obs[0]]

        for t in range(1, T):
            for i in range(N):
                for j in range(N):
                    alpha[t][i] += alpha[t - 1][j] * A[j][i] * B[i][obs[t]]

        out = sum(alpha[T - 1])
        return out
