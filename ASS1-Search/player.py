#!/usr/bin/env python3
import random

from fishing_game_core.game_tree import Node, State
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR
from math import sqrt
GRID_DIM = 20
DEPTH = 3
ADD_TERM = 0.1


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

        children = initial_tree_node.compute_and_get_children()

        max_score = float("-inf")
        max_index = -1

        for i,child in enumerate(children):
            score = self.alpha_beta_pruning(child, DEPTH-1, max_score, float('+inf'), 0)
            if score > max_score:
                max_score = score
                max_index = i

        return ACTION_TO_STR[max_index]

    def alpha_beta_pruning(self, node, depth, alpha, beta, maximizing_player):

        if depth == 0:
            return self.heuristicScore(node)

        if maximizing_player:
            max_eval = float('-inf')
            for child in node.compute_and_get_children():
                eval = self.alpha_beta_pruning(child, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval

        else:
            min_eval = float('inf')
            for child in node.compute_and_get_children():
                eval = self.alpha_beta_pruning(child, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval

    def heuristicScore(self, node):

        score = 0
        player = node.state.get_player()
        my_position = node.state.get_hook_positions()[player]

        if player == 0:
            opponent_position = node.state.get_hook_positions()[1]
        else:
            opponent_position = node.state.get_hook_positions()[0]

        fish_positions = node.state.get_fish_positions()
        fish_scores = node.state.get_fish_scores()

        for fish in fish_positions:
            my_score = fish_scores[fish]/(self.distance(my_position,fish_positions[fish])+ADD_TERM)
            opponent_score = fish_scores[fish]/(self.distance(opponent_position,fish_positions[fish])+ADD_TERM)
            score += my_score-opponent_score
        score += node.state.get_player_scores()[0]-node.state.get_player_scores()[1]
        return score

    def distance(self, hook_position, fish_position, method='n_moves'):

        x_dist = min((hook_position[0] - fish_position[0]) % GRID_DIM, (fish_position[0] - hook_position[0]) % GRID_DIM)
        y_dist = abs(fish_position[1] - hook_position[1])

        if method == 'n_moves':
            return x_dist + y_dist
        elif method == 'euclidean':
            return sqrt(x_dist ** 2 + y_dist ** 2)

