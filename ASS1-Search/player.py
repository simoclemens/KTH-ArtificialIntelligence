#!/usr/bin/env python3
import random

from fishing_game_core.game_tree import Node, State
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR
from math import sqrt
GRID_DIM = 20
DEPTH = 4
ADD_TERM = 1


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

        ordered_children = sorted(children, key=lambda x: self.heuristicScore(x),reverse=True)

        max_score = float("-inf")
        max_index = -1

        for i,child in enumerate(ordered_children):
            score = self.alpha_beta_pruning(child, DEPTH-1, max_score, float('+inf'), False)
            if score > max_score:
                max_score = score
                max_index = i
        best_move = ordered_children[max_index].move

        return ACTION_TO_STR[best_move]

    def alpha_beta_pruning(self, node, depth, alpha, beta, maximizing_player):

        children = node.compute_and_get_children()

        if depth == 0 or len(children) == 0:
            return self.heuristicScore(node)

        if maximizing_player:
            v = float('-inf')
            ordered_children = sorted(children, key=lambda x: self.heuristicScore(x),reverse=True)
            for child in ordered_children:
                v = max(v,self.alpha_beta_pruning(child, depth - 1, alpha, beta, False))
                alpha = max(alpha, v)
                if beta <= alpha:
                    break  # Beta cutoff
            return alpha

        else:
            v = float('inf')
            ordered_children = sorted(children, key=lambda x: self.heuristicScore(x), reverse=False)
            for child in ordered_children:
                v = min(v, self.alpha_beta_pruning(child, depth - 1, alpha, beta, True))
                beta = min(beta, v)
                if beta <= alpha:
                    break  # Alpha cutoff
            return beta

    def heuristicScore(self, node):

        score = 0
        my_position = node.state.get_hook_positions()[0]
        opponent_position = node.state.get_hook_positions()[1]

        fish_positions = node.state.get_fish_positions()
        fish_scores = node.state.get_fish_scores()

        for fish in fish_positions:
            my_dist = self.distance(my_position,fish_positions[fish])
            opponent_dist = self.distance(opponent_position, fish_positions[fish])
            score += fish_scores[fish]/(my_dist+ADD_TERM) - 0.25*fish_scores[fish]/(opponent_dist + ADD_TERM)
        score += node.state.get_player_scores()[0]-node.state.get_player_scores()[1]

        return score

    def distance(self, hook_position, fish_position, method='n_moves'):

        x_dist = min((hook_position[0] - fish_position[0]) % GRID_DIM, (fish_position[0] - hook_position[0]) % GRID_DIM)
        y_dist = abs(fish_position[1] - hook_position[1])

        if method == 'n_moves':
            return x_dist + y_dist
        elif method == 'euclidean':
            return sqrt(x_dist ** 2 + y_dist ** 2)



