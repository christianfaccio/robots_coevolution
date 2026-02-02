import os
import neat
import pickle
from tqdm import tqdm
from game.game import Game 

def match(game, net1, net2):
    """
    Run a match and return (winner, energy1, energy2).
    winner: 1 if player 1 wins, 2 if player 2 wins, 0 for draw
    """
    game.reset()
    while not game.game_over:
        action1 = net1.activate(game.robot1.state(game.robot2))
        res1 = action1.index(max(action1))
        action2 = net2.activate(game.robot2.state(game.robot1))
        res2 = action2.index(max(action2))
        game.play(res1, res2)
        game.check_game_over()

    energy1 = game.robot1.energy
    energy2 = game.robot2.energy

    if game.winner == 1:
        return 1, energy1, energy2
    elif game.winner == 2:
        return 2, energy1, energy2
    else:
        return 0, energy1, energy2

def eval_genomes(genomes, config):
    game = Game(render=False)
    ge = []
    nets = []
    for _, genome in genomes: # population
        ge.append(genome)
        nets.append(neat.nn.FeedForwardNetwork.create(genome, config))
        genome.fitness = 0.0

    # Round-robin tournament
    n = len(ge)
    total_matches = n * (n - 1) // 2
    with tqdm(total=total_matches, desc="Round-robin", leave=False) as pbar:
        for i in range(n):
            for j in range(i + 1, n):  # Only play each pair once
                result, energy1, energy2 = match(game, nets[i], nets[j])

                # Base points for win/draw/loss
                if result == 1:
                    ge[i].fitness += 3  # Player 1 wins
                elif result == 2:
                    ge[j].fitness += 3  # Player 2 wins
                else:
                    ge[i].fitness += 1  # Draw
                    ge[j].fitness += 1

                # Bonus for energy collected (encourages active play)
                # Scale: 100 energy = 1 fitness point
                ge[i].fitness += energy1 / 100.0
                ge[j].fitness += energy2 / 100.0

                pbar.update(1)

    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.txt', help='Path to NEAT config file')
    parser.add_argument('--generations', type=int, default=100, help='Number of generations to train')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file to resume from')
    args = parser.parse_args()

    # Load configuration
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        args.config
    )

    if args.checkpoint:
        # Resume from checkpoint
        print(f"Resuming from checkpoint: {args.checkpoint}")
        pop = neat.Checkpointer.restore_checkpoint(args.checkpoint)
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

    checkpointer = neat.Checkpointer(
        generation_interval=10,
        filename_prefix='checkpoints/neat-checkpoint-'
    )
    pop.add_reporter(checkpointer)

    pop.run(eval_genomes, args.generations)

    with open('best_genome.pkl', 'wb') as f:
        pickle.dump(stats.best_genome(), f)