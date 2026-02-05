import os
import copy
import neat
import pickle
from tqdm import tqdm
from game.game import Game


class HallOfFame:
    """
    Maintains a collection of the best genomes from past generations.
    This prevents 'forgetting' by ensuring new strategies must still beat historical champions.
    """
    def __init__(self, max_size=10):
        self.max_size = max_size
        self.members = []  # List of (genome, network) tuples
        self.generation_added = []  # Track when each member was added

    def update(self, genomes, config, current_generation):
        """
        Add the best genome from the current generation to the hall of fame.
        Uses fitness diversity to avoid adding too similar individuals.
        """
        if not genomes:
            return

        # Find the best genome
        best_genome = max(genomes, key=lambda g: g[1].fitness if g[1].fitness else 0)
        _, genome = best_genome

        if genome.fitness is None:
            return

        # Create a deep copy to preserve the genome state
        genome_copy = copy.deepcopy(genome)
        net = neat.nn.FeedForwardNetwork.create(genome_copy, config)

        # Add to hall of fame
        self.members.append((genome_copy, net))
        self.generation_added.append(current_generation)

        # If over capacity, remove oldest (or weakest) member
        if len(self.members) > self.max_size:
            # Remove the oldest member (FIFO strategy)
            self.members.pop(0)
            self.generation_added.pop(0)

    def get_networks(self):
        """Return list of neural networks for hall of fame members."""
        return [net for _, net in self.members]

    def __len__(self):
        return len(self.members)


# Global hall of fame instance
hall_of_fame = HallOfFame(max_size=10)


def match(game, net1, net2):
    """
    Run a match and return (winner, energy1, energy2, collision_kill).
    winner: 1 if player 1 wins, 2 if player 2 wins, 0 for draw
    collision_kill: True if game ended by collision
    """
    game.reset()
    while not game.game_over:
        output1 = net1.activate(game.robot1.state(game.robot2))
        output2 = net2.activate(game.robot2.state(game.robot1))
        game.play(output1, output2)
        game.check_game_over()

    energy1 = game.robot1.energy
    energy2 = game.robot2.energy
    collision_kill = game.collision_kill

    if game.winner == 1:
        return 1, energy1, energy2, collision_kill
    elif game.winner == 2:
        return 2, energy1, energy2, collision_kill
    else:
        return 0, energy1, energy2, collision_kill


def eval_genomes(genomes, config):
    game = Game(render=False)
    ge = []
    nets = []
    for _, genome in genomes:
        ge.append(genome)
        nets.append(neat.nn.FeedForwardNetwork.create(genome, config))
        genome.fitness = 0.0

    n = len(ge)
    hof_nets = hall_of_fame.get_networks()
    hof_size = len(hof_nets)

    # Calculate total matches: round-robin + hall of fame matches
    round_robin_matches = n * (n - 1) // 2
    hof_matches = n * hof_size
    total_matches = round_robin_matches + hof_matches

    collision_bonus = 5  # Extra reward for winning by collision (aggressive play)
    hof_weight = 2.0  # Hall of fame matches are worth more (to maintain pressure)

    with tqdm(total=total_matches, desc="Tournament", leave=False) as pbar:
        # Round-robin tournament within current population
        for i in range(n):
            for j in range(i + 1, n):
                result, energy1, energy2, collision_kill = match(game, nets[i], nets[j])

                if result == 1:
                    ge[i].fitness += 10
                    if collision_kill:
                        ge[i].fitness += collision_bonus
                elif result == 2:
                    ge[j].fitness += 10
                    if collision_kill:
                        ge[j].fitness += collision_bonus
                else:
                    ge[i].fitness += 1
                    ge[j].fitness += 1

                ge[i].fitness += energy1
                ge[j].fitness += energy2

                pbar.update(1)

        # Play against Hall of Fame members
        if hof_size > 0:
            for i in range(n):
                for hof_net in hof_nets:
                    result, energy1, energy2, collision_kill = match(game, nets[i], hof_net)

                    # Weighted rewards for hall of fame matches
                    if result == 1:
                        ge[i].fitness += 10 * hof_weight
                        if collision_kill:
                            ge[i].fitness += collision_bonus * hof_weight
                    elif result == 2:
                        # Lost to hall of fame - no penalty, just no reward
                        pass
                    else:
                        ge[i].fitness += 1 * hof_weight

                    # Energy bonus (still counts but not weighted)
                    ge[i].fitness += energy1

                    pbar.update(1)


class HallOfFameReporter(neat.reporting.BaseReporter):
    """Reporter that updates the Hall of Fame after each generation."""

    def __init__(self, hof, config):
        self.hall_of_fame = hof
        self.config = config
        self.current_generation = 0

    def start_generation(self, generation):
        self.current_generation = generation

    def post_evaluate(self, config, population, species_set, best_genome):
        # Convert population dict to list format expected by HallOfFame
        genomes = [(gid, g) for gid, g in population.items()]
        self.hall_of_fame.update(genomes, self.config, self.current_generation)
        print(f"  Hall of Fame size: {len(self.hall_of_fame)}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.txt', help='Path to NEAT config file')
    parser.add_argument('--generations', type=int, default=100, help='Number of generations to train')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file to resume from')
    parser.add_argument('--hof-size', type=int, default=10, help='Hall of Fame size (max champions to keep)')
    args = parser.parse_args()

    # Load configuration
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        args.config
    )

    # Initialize hall of fame with specified size
    hall_of_fame = HallOfFame(max_size=args.hof_size)

    if args.checkpoint:
        # Resume from checkpoint
        print(f"Resuming from checkpoint: {args.checkpoint}")
        pop = neat.Checkpointer.restore_checkpoint(args.checkpoint)

        # Try to load hall of fame if it exists
        hof_path = 'checkpoints/hall_of_fame.pkl'
        if os.path.exists(hof_path):
            with open(hof_path, 'rb') as f:
                hof_data = pickle.load(f)
                # Rebuild networks for loaded genomes
                for genome, gen_added in zip(hof_data['genomes'], hof_data['generations']):
                    net = neat.nn.FeedForwardNetwork.create(genome, config)
                    hall_of_fame.members.append((genome, net))
                    hall_of_fame.generation_added.append(gen_added)
            print(f"Loaded Hall of Fame with {len(hall_of_fame)} members")
    else:
        # Start fresh - clear previous checkpoints
        if os.path.exists('checkpoints'):
            for file in os.listdir('checkpoints'):
                file_path = os.path.join('checkpoints', file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        else:
            os.makedirs('checkpoints')
        pop = neat.Population(config)

    # Add reporters
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    # Add Hall of Fame reporter
    hof_reporter = HallOfFameReporter(hall_of_fame, config)
    pop.add_reporter(hof_reporter)

    checkpointer = neat.Checkpointer(
        generation_interval=10,
        filename_prefix='checkpoints/neat-checkpoint-'
    )
    pop.add_reporter(checkpointer)

    try:
        pop.run(eval_genomes, args.generations)
    finally:
        # Save hall of fame on exit
        hof_path = 'checkpoints/hall_of_fame.pkl'
        hof_data = {
            'genomes': [genome for genome, _ in hall_of_fame.members],
            'generations': hall_of_fame.generation_added
        }
        with open(hof_path, 'wb') as f:
            pickle.dump(hof_data, f)
        print(f"Saved Hall of Fame with {len(hall_of_fame)} members to {hof_path}")

    with open('best_genome.pkl', 'wb') as f:
        pickle.dump(stats.best_genome(), f)
