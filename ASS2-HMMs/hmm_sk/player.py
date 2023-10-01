#!/usr/bin/env python3

from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import random
import math
import numpy as np

def min_max_scaling(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    scaled_matrix = (matrix - min_val) / (max_val - min_val)
    return scaled_matrix

def flatten_matrix(matrix):
    return [item for sublist in matrix for item in sublist]


class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def __init__(self):
        super().__init__()
        self.revealed_fish = None
        self.probabilities = None
        self.observations = None
        self.models = None
        self.guesses = None

    def init_parameters(self):
        models = 0
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """

        self.n_fish_types = 7
        self.n_fishes = 70

        # init one model for each fish type
        self.models = [HMMModel() for _ in range(self.n_fish_types)]

        # create the observations matrix, with one line for each type of fish
        self.observations = [[] for _ in range(self.n_fishes)]

        # create probabilities matrix, with one row per type and one column per fish
        self.probabilities = np.random.rand(self.n_fish_types, self.n_fishes).tolist()

        # create a list of seven lists, each one containing the fishes of the given type
        self.revealed_fish = [[] for _ in range(self.n_fish_types)]

        self.correctly_guessed = [[] for _ in range(self.n_fish_types)] 

        # keeps track of the guesses made
        self.guesses = []

    def fish_obs_seqs(self):
        # observation sequences for all fish
        transposed_observations = [list(col) for col in zip(*self.observations)]
        return transposed_observations

    def insert_obs(self, observations):
        for i, obs in enumerate(observations):
            self.observations[i].append(obs)

    def argmax_matrix(self, ignore_correctly_guessed=False, ignore_revealed=False, ignore_guessed=False):
        max_prob = -1
        coord = (None, None)

        # j is fishes, i is fish types
        for i, seq in enumerate(self.probabilities):
            for j, prob in enumerate(seq): 
                if ignore_correctly_guessed and j in flatten_matrix(self.correctly_guessed):
                    continue
                # don't guess revealed fish
                if ignore_revealed and j in flatten_matrix(self.revealed_fish):
                    continue
                # don't repeat guesses
                if self.guesses and (i, j) in self.guesses:
                    continue
                if prob > max_prob:
                    max_prob = prob
                    coord = (i, j)

        return coord
    
    def revealed_but_incorrect(self):
        for fish_type, fish_ids in enumerate(self.revealed_fish):
            for fish_id in fish_ids:
                if not fish_id in flatten_matrix(self.correctly_guessed):
                    return (fish_id, fish_type)

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

        # update probabilities
        for i, obs in enumerate(self.observations):
            for j, m in enumerate(self.models):
                self.probabilities[j][i] = m.compute_seq_prob(obs)
        
        # normalize
        #self.probabilities = min_max_scaling(self.probabilities)
        #print(self.probabilities)

        fish_type, fish_id = self.argmax_matrix(ignore_correctly_guessed=True, ignore_revealed=True, ignore_guessed=True)
        
        # if revealed all fish already
        if fish_type is None:
            guess = self.revealed_but_incorrect()
            if not guess:
                return None
            (fish_id, fish_type) = guess
        
        prob = self.probabilities[fish_type][fish_id]

        guess = (fish_id, fish_type)
        print("Guess: ", guess, " with probability: ", prob)

        self.guesses.append(guess)

        return guess

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
        #self.guesses[-1] += (correct,)
        if correct: self.correctly_guessed[true_type].append(fish_id)

        self.revealed_fish[true_type].append(fish_id)
        print("Revealed fish: ", self.revealed_fish)

        # train models
        self.models = [HMMModel() for _ in range(self.n_fish_types)]

        if len(self.observations[0]) > 2:
            for i, m in enumerate(self.models):
                for index in self.revealed_fish[i]:
                    m.train(self.observations[index])


def normalize(matrix, axis=1):
    matrix /= matrix.sum(axis=axis, keepdims=True)
    return matrix


class HMMModel:
    def __init__(self, hidden_states=2, n_obs=8):
        self.A = normalize(np.random.rand(hidden_states, hidden_states)).tolist()
        self.B = normalize(np.random.rand(hidden_states, n_obs)).tolist()
        self.p = normalize(np.random.rand(hidden_states), axis=0).tolist()

    def train(self, obs):
        EPSILON = 1e-10

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
            alpha = np.full((T, N), -np.inf)  # Initialize to -inf for log probabilities
            c_log = []
            c = []

            # first row
            for i in range(N):
                alpha[0][i] = np.log(p0[i]) + np.log(B[i][obs[0]])

            c_t_log = np.logaddexp.reduce(alpha[0])  # Calculate c_t in log domain

            for i in range(N):
                alpha[0][i] -= c_t_log

            c_log.append(c_t_log)

            # computation
            for t in range(1, T):
                for i in range(N):
                    alpha[t][i] = np.logaddexp.reduce(
                        [alpha[t - 1][j] + np.log(A[j][i]) for j in range(N)]
                    ) + np.log(B[i][obs[t]])

                c_t_log = np.logaddexp.reduce(alpha[t])  # Calculate c_t in log domain

                for i in range(N):
                    alpha[t][i] -= c_t_log

                c_log.append(c_t_log)

            mle_new = -np.sum(c_log)

            # BETA
            # initialization
            beta = np.full((T, N), -np.inf)  # Initialize to -inf for log probabilities

            # first row
            for i in range(N):
                beta[T - 1][i] = 0  # In log domain, 0 is equivalent to 1 (probability)

            # computation
            for t in range(T - 2, -1, -1):
                for i in range(N):
                    beta[t][i] = np.logaddexp.reduce(
                        [beta[t + 1][j] + np.log(B[j][obs[t + 1]]) + np.log(A[i][j]) for j in range(N)]
                    )

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

                    A[i][j] = num / (den + EPSILON)

            # B update
            for i in range(N):
                for k in range(K):
                    den = 0
                    num = 0
                    for t in range(T):
                        den += gamma[t][i]
                        if obs[t] == k:
                            num += gamma[t][i]

                    B[i][k] = num / (den + EPSILON)

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
