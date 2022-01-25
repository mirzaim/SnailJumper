import copy
import random

import numpy as np

from player import Player
from variables import global_variables

def write_records(min, mean, max):
    with open(global_variables['log_file'], 'a') as f:
        f.write(f'{min}, {mean}, {max}\n')

class Evolution:
    def __init__(self):
        self.game_mode = "Neuroevolution"

    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """
        fitnesses = [player.fitness for player in players]
        fitnesses_avg = sum(fitnesses) // len(fitnesses)

        # TODO (Implement top-k algorithm here)
        # players = sorted(players, key=lambda x: x.fitness, reverse=True)

        # TODO (Additional: Implement roulette wheel here)
        # players = random.choices(players, weights=fitnesses, k=num_players)

        # TODO (Additional: Implement SUS here)
        # ac_fitnesses = np.add.accumulate(fitnesses)
        # selected_elms = np.vectorize(lambda x: np.argmax(ac_fitnesses > x))(
        #     np.add.accumulate([fitnesses_avg] * num_players) - np.random.randint(fitnesses_avg))
        # players = [players[i] for i in selected_elms]

        # TODO (Additional: Implement Q-tournament here)
        players = [max(random.sample(players, 5), key=lambda x: x.fitness) for _ in range(num_players)]
        

        # TODO (Additional: Learning curve)
        write_records(min(fitnesses), sum(fitnesses)/len(fitnesses), max(fitnesses))

        return players[: num_players]

    def generate_new_population(self, num_players, prev_players=None):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """
        first_generation = prev_players is None
        if first_generation:
            return [Player(self.game_mode) for _ in range(num_players)]
        else:
            # TODO ( Parent selection and child generation )
            # new_players = prev_players  # DELETE THIS AFTER YOUR IMPLEMENTATION

            # random.shuffle(prev_players)
            fitnesses = [player.fitness for player in prev_players]
            prev_players = random.choices(prev_players, weights=fitnesses, k=num_players)

            new_players = []
            for father, mother in zip(prev_players, iter(prev_players)):
                father, mother = self.clone_player(father), self.clone_player(mother)
                child1, child2 = Player(self.game_mode), Player(self.game_mode)
                for i in range(1, father.nn.n + 1):
                    # crossover
                    mask = np.random.choice([True, False], size=father.nn.params[f'b{i}'].shape[0])
                    child1.nn.params[f'W{i}'] = mask * father.nn.params[f'W{i}'] + (1-mask) * mother.nn.params[f'W{i}']
                    child2.nn.params[f'W{i}'] = mask * mother.nn.params[f'W{i}'] + (1-mask) * father.nn.params[f'W{i}']
                    child1.nn.params[f'b{i}'] = mask * father.nn.params[f'b{i}'] + (1-mask) * mother.nn.params[f'b{i}']
                    child2.nn.params[f'b{i}'] = mask * mother.nn.params[f'b{i}'] + (1-mask) * father.nn.params[f'b{i}']

                    # mutation
                    if np.random.rand() < 0.15:
                        child1.nn.params[f'W{i}'] = np.random.normal(size=child1.nn.params[f'W{i}'].shape)
                        child1.nn.params[f'b{i}'] = np.random.normal(size=child1.nn.params[f'b{i}'].shape)
                    if np.random.rand() < 0.15:
                        child2.nn.params[f'W{i}'] = np.random.normal(size=child2.nn.params[f'W{i}'].shape)
                        child2.nn.params[f'b{i}'] = np.random.normal(size=child2.nn.params[f'b{i}'].shape)

                new_players.extend((child1, child2))              

            return new_players

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player
