import os
import math
import pickle
import neat
from tqdm import tqdm
from game.game import Game, MAX_OBSTACLES

# TODO: review
def state_to_input(state):
    """Convert game state dict to fixed-size neural network input.

    State dict contains:
        - x, y, angle: robot position/orientation (normalized)
        - enemy_distance: distance to enemy (can be None if out of range)
        - angle_to_enemy: relative angle to enemy (normalized)
        - enemy_health, self_health: health values (normalized)
        - time_to_shoot: cooldown (normalized)
        - walls_distances: [S, E, N, W] distances (can be None)
        - obstacles_distances: list of (id, distance) tuples

    Returns fixed-size list of floats for neural network.
    Total inputs: 8 basic + 4 walls + 8 obstacles = 20
    """
    inputs = []

    # Basic state (8 values)
    inputs.append(state['x'])
    inputs.append(state['y'])
    inputs.append(state['angle'])
    inputs.append(state['enemy_distance'])
    inputs.append(state['angle_to_enemy'])
    inputs.append(state['enemy_health'])
    inputs.append(state['self_health'])
    inputs.append(state['time_to_shoot'])

    # Wall distances (4 values)
    for wall_dist in state['walls_distances']:
        inputs.append(wall_dist)

    # Obstacle distances 
    obstacle_dists = [dist for _, dist in state['obstacles_distances']]

    for i in range(MAX_OBSTACLES):
        if i < len(obstacle_dists):
            inputs.append(obstacle_dists[i])
        else:
            inputs.append(1.0)  # No obstacle = max distance

    return inputs


def genome_to_action(net, state):
    """Convert neural network output to game action.

    Outputs (3):
        - movement rotation (-1 to 1, scaled to -pi to pi)
        - turret rotation (-1 to 1, scaled to -pi to pi)
        - shoot (threshold at 0)
    """
    inputs = state_to_input(state)
    output = net.activate(inputs)

    return {
        'movement': output[0] * math.pi,  # tanh output is -1 to 1
        'turret': output[1] * math.pi,
        'shoot': output[2] > 0  # tanh: positive = shoot
    }


def run_match(net1, net2):
    """Run a single match between two neural networks.

    Returns:
        (score1, score2): fitness scores for each network
    """
    game = Game(render=False)

    while not game.game_over:
        state = game.get_state()

        # Get actions from neural networks
        action1 = genome_to_action(net1, state['robot1'])
        action2 = genome_to_action(net2, state['robot2'])

        # Step the game
        game.step(action1, action2)

    # Calculate fitness scores (for now based on winning) -> TODO: implement other fitness measures
    if game.winner == 1:
        return 1,0
    elif game.winner == 2:
        return 0,1
    else:
        return 0.5,0.5


def eval_genomes(genomes, config):
    """Evaluate all genomes using round-robin tournament.

    Each genome plays against every other genome in the population.
    Fitness is the average score across all matches.
    """
    # Create neural networks for all genomes
    nets = {}
    for genome_id, genome in genomes:
        nets[genome_id] = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0.0  # Initialize fitness

    genome_list = list(genomes)
    n = len(genome_list)

    if n < 2:
        # Not enough genomes for tournament, give default fitness
        for genome_id, genome in genomes:
            genome.fitness = 1.0
        return

    # Round-robin: each genome plays against every other genome
    total_matches = n * (n - 1) // 2
    match_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    for i, j in tqdm(match_pairs, total=total_matches, desc="Matches", leave=False):
        genome_id_1, genome_1 = genome_list[i]
        genome_id_2, genome_2 = genome_list[j]

        net1 = nets[genome_id_1]
        net2 = nets[genome_id_2]

        # Run match
        score1, score2 = run_match(net1, net2)

        # Accumulate fitness
        genome_1.fitness += score1
        genome_2.fitness += score2

    # Normalize fitness by number of matches played per genome
    matches_per_genome = n - 1
    for genome_id, genome in genomes:
        genome.fitness /= matches_per_genome


def run_training(config_path, n_generations=100, checkpoint_interval=10):
    """Run NEAT training with round-robin tournament."""

    # Load configuration
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    # Create population
    pop = neat.Population(config)

    # Add reporters for output
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    # Add checkpointer to save progress
    checkpointer = neat.Checkpointer(
        generation_interval=checkpoint_interval,
        filename_prefix='checkpoints/neat-checkpoint-'
    )
    pop.add_reporter(checkpointer)

    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)

    # Run evolution
    winner = pop.run(eval_genomes, n_generations)

    # Save the best genome
    with open('best_genome.pkl', 'wb') as f:
        pickle.dump(winner, f)

    return winner, stats


def resume_training(checkpoint_path, n_generations=100):
    """Resume training from a checkpoint."""

    # Restore from checkpoint
    pop = neat.Checkpointer.restore_checkpoint(checkpoint_path)

    # Re-add reporters
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    checkpointer = neat.Checkpointer(
        generation_interval=10,
        filename_prefix='checkpoints/neat-checkpoint-'
    )
    pop.add_reporter(checkpointer)

    # Continue evolution
    winner = pop.run(eval_genomes, n_generations)

    with open('best_genome.pkl', 'wb') as f:
        pickle.dump(winner, f)

    return winner, stats


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train robots with NEAT')
    parser.add_argument('--mode', choices=['train', 'resume'], default='train',
                        help='Mode: train or resume')
    parser.add_argument('--generations', type=int, default=100,
                        help='Number of generations to train')
    parser.add_argument('--genome', type=str, default='best_genome.pkl',
                        help='Path to genome file for play mode')
    parser.add_argument('--checkpoint', type=str,
                        help='Checkpoint file for resume mode')
    parser.add_argument('--config', type=str, default='config.txt',
                        help='Path to NEAT config file')

    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), args.config)

    if args.mode == 'train':
        print("------ Starting NEAT training with round-robin tournament... ------")
        winner, stats = run_training(config_path, args.generations)
        print('------ Finished training. Best genome saved to best_genome.pkl ------')

    elif args.mode == 'resume':
        if not args.checkpoint:
            print("Please specify checkpoint file with --checkpoint")
        else:
            print(f"------ Resuming from {args.checkpoint} ------")
            winner, stats = resume_training(args.checkpoint, args.generations)
            print('------ Finished training. Best genome saved to best_genome.pkl ------')
