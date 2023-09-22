#!/usr/bin/env python3

from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import random
import math


class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """
        pass

    def guess(self, step, observations):
        """
        This method gets called on every iteration, providing observations.
        Here the player should process and store this information,
        and optionally make a guess by returning a tuple containing the fish index and the guess.
        :param step: iteration number
        :param observations: a list of N_FISH observations, encoded as integers
        :return: None or a tuple (fish_id, fish_type)
        """

        # This code would make a random guess on each step:
        return step % N_FISH, random.randint(0, N_SPECIES - 1)

        # return None

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
        pass

    def trainModel(self, A, B, p0, obs):

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

        return A, B, p0
