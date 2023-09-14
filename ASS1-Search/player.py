#!/usr/bin/env python3
import random

from fishing_game_core.game_tree import Node, State
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR
from math import sqrt, e, exp

import time

GRID_DIM = 20
DEPTH = 10
W1 = 2.5
W2 = 0.5


class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and send
        this to the game through the sender. Then it receives an
        update of the game through receiver, with this it computes the
        next movement.
        :return:
        """

        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return


class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()
        self.transposition_table = {}

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """

        # Generate first message (Do not remove this line!)
        first_msg = self.receiver()

        while True:
            msg = self.receiver()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move(initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    def start_clock(self):
        self.start_time = time.time()

    def time_elapsed(self):
        return time.time() - self.start_time

    def is_time_up(self: float):

        is_up = self.time_elapsed() >= self.cutoff_time * 0.001
        # if is_up: print('times_up')
        return is_up

    def search_best_next_move(self, initial_tree_node):
        """
        Use minimax (and extensions) to find best possible next move for player 0 (green boat)
        :param initial_tree_node: Initial game tree node
        :type initial_tree_node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """
        # EDIT THIS METHOD TO RETURN BEST NEXT POSSIBLE MODE USING MINIMAX ###

        # NOTE: Don't forget to initialize the children of the current node
        #       with its compute_and_get_children() method!

        self.start_clock()
        self.cutoff_time = 59

        index = self.ids(initial_tree_node)
        best_move = self.order_children(initial_tree_node)[index].move

        return ACTION_TO_STR[best_move]

    def order_children(self, node, is_max=True):
        children = node.compute_and_get_children()
        return sorted(children, key=lambda x: self.heuristicScore(x), reverse=is_max)

    def ids(self, node):

        # transposition_table = {}

        # iterative deepening search
        max_score = float("-inf")
        max_index = -1

        for depth in range(1, DEPTH + 1):
            score, index = self.dls(node, depth, self.transposition_table)
            if score > max_score:
                max_score = score
                max_index = index

        return max_index

    def dls(self, node, depth, transposition_table):
        max_score = float("-inf")
        max_index = -1

        if not depth == 0 or not self.is_time_up():
            children = node.compute_and_get_children()

            ordered_children = sorted(children, key=lambda x: self.heuristicScore(x), reverse=True)

            for i, child in enumerate(ordered_children):
                if self.is_time_up():
                    return max_score, max_index

                score = self.alpha_beta_pruning(child, depth, max_score, float('+inf'), False, transposition_table)
                if score > max_score:
                    max_score = score
                    max_index = i

        return max_score, max_index

    def alpha_beta_pruning(self, node, depth, alpha, beta, maximizing_player, transposition_table):

        h = self.get_hash_state(node)

        if h in transposition_table \
                and depth <= transposition_table[h][0]:
            return transposition_table[h][1]

        children = node.compute_and_get_children()

        if depth == 0 or len(children) == 0 or self.is_time_up():
            score = self.heuristicScore(node)
            transposition_table[h] = (depth, score)
            return score

        if maximizing_player:
            v = float('-inf')
            ordered_children = sorted(children, key=lambda x: self.heuristicScore(x), reverse=True)
            for child in ordered_children:
                if self.is_time_up():
                    return self.heuristicScore(node)

                v = max(v, self.alpha_beta_pruning(child, depth - 1, alpha, beta, False, transposition_table))
                alpha = max(alpha, beta)

                if beta <= alpha:
                    break  # Beta cutoff

        else:
            v = float('+inf')
            ordered_children = sorted(children, key=lambda x: self.heuristicScore(x), reverse=False)
            for child in ordered_children:
                if self.is_time_up():
                    return self.heuristicScore(node)

                v = min(v, self.alpha_beta_pruning(child, depth - 1, alpha, beta, True, transposition_table))
                beta = min(beta, v)

                if beta <= alpha:
                    break  # Alpha cutoff

        transposition_table.update({h:(depth,v)})
        return v

    def heuristicScore(self, node):

        score = 0
        state = node.state
        my_position = node.state.get_hook_positions()[0]
        opponent_position = node.state.get_hook_positions()[1]

        h = 0
        for i, coord in state.fish_positions.items():
            h = max(h, state.fish_scores[i] * (exp(-self.distance(my_position, coord) - W2 *exp(-self.distance(opponent_position, coord)))))

        score += node.state.get_player_scores()[0] - node.state.get_player_scores()[1]

        return W1 * score + h

    def distance(self, hook_position, fish_position, method='n_moves'):

        x_dist = min((hook_position[0] - fish_position[0]) % GRID_DIM, (fish_position[0] - hook_position[0]) % GRID_DIM)
        y_dist = abs(fish_position[1] - hook_position[1])

        if method == 'n_moves':
            return x_dist + y_dist
        elif method == 'euclidean':
            return sqrt(x_dist ** 2 + y_dist ** 2)

    def get_hash_state(self, node):
        state = node.state
        fish_pos = state.get_fish_positions()
        hook_pos = state.get_hook_positions()
        fish_score = state.get_fish_scores()
        composite_key = {
            "{:02d}{:02d}".format(fish_pos[0], fish_pos[1]): fish_score[fish_idx]
            for fish_idx, fish_pos in fish_pos.items()
        }
        return str(hook_pos) + str(composite_key)