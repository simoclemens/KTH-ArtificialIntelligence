#!/usr/bin/env python3
from player_controller_hmm import PlayerControllerHMMAbstract
import math
import numpy as np


class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def __init__(self):
        super().__init__()
        self.observations = None
        self.models = None
        self.last_obs = None

    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """
        # init one model for each fish type
        self.models = [HMMModel(i) for i in range(7)]

        # create the observations matrix, with one line for each type of fish
        self.observations = [[] for _ in range(70)]

    def insert_obs(self, observations):
        for i in range(len(self.observations)):
            self.observations[i].append(observations[i])

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

        if step < 105:
            return None

        else:
            self.last_obs = self.observations.pop()
            fish_id = len(self.observations)
            max_prob = -1
            fish_type = None
            for i, m in enumerate(self.models):
                prob = m.compute_seq_prob(self.last_obs)
                if prob > max_prob:
                    max_prob = prob
                    fish_type = i

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
        if not correct:
            self.models[true_type].train(self.last_obs)


def generate_matrix(r, c):
    M = [[(1 / c) + np.random.rand() / 1000 for _ in range(c)] for _ in range(r)]
    for i, row in enumerate(M):
        div = 1/sum(row)
        for j in range(c):
            M[i][j] *= div
    return M


class HMMModel:
    def __init__(self, fish_type, hidden_states=2, n_obs=8):

        self.fish_type = fish_type
        self.A = generate_matrix(hidden_states, hidden_states)
        self.B = generate_matrix(hidden_states, n_obs)
        self.p = generate_matrix(1, hidden_states)[0]
        self.epsilon = 1e-15

    def train(self, obs):

        # class -> alg
        A = self.A
        B = self.B
        p0 = self.p

        T = len(obs)
        N = len(A)
        K = len(B[0])
        MAX_ITER = 10

        mle = float("-inf")
        mle_new = float("-inf")
        iteration = 0

        while (mle_new > mle and iteration < MAX_ITER) or mle == float("-inf"):
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

            c_t = 1 / (sum(alpha[0])+self.epsilon)

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

                c_t = 1 / (sum(alpha[t])+self.epsilon)

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
                    gamma[t][i] = sum(di_gamma[t][i])

            # last row
            gamma[T - 1] = alpha[T - 1]

            # initial state update
            p0 = gamma[0]

            # A update
            for i in range(N):
                den = 0
                for t in range(T - 1):
                    den += gamma[t][i]
                den += self.epsilon
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
                    den += self.epsilon
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
