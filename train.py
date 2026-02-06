import os
import copy
import neat
import random
import pickle
from tqdm import tqdm
from game.game import Game

def match(game, net1, net2, inv=False, extra_energy_1=None, extra_energy_2=None):
    """
    Run a match and return the points for the HOST genome:
    - 1 point for victory
    - 0 points for defeat
    - 0 points if the game ends by timeout

    inv flag indicates if the host genome is playing as robot1 (inv=False) or robot2 (inv=True)
    """
    game.reset(extra_energy_1=extra_energy_1, extra_energy_2=extra_energy_2)
    net1.reset()
    net2.reset()
    while not game.game_over:
        if not inv:
            output1 = net1.activate(game.robot1.state(game.robot2))
            output2 = net2.activate(game.robot2.state(game.robot1))
        else:
            output1 = net2.activate(game.robot1.state(game.robot2))
            output2 = net1.activate(game.robot2.state(game.robot1))
        game.play(output1, output2)

    if game.winner == 1:
        return (1, 0) if not inv else (0, 1)
    elif game.winner == 2:
        return (0, 1) if not inv else (1, 0)
    else:
        return (0, 0)
    
def get_species_champions(pop, n=4):
    champions = []
    for species in pop.species.species.values():
        # Skip empty species
        if not species.members:
            continue
        # Find best genome in this species
        champion = max(species.members.values(), key=lambda g: g.fitness or 0)
        champions.append(champion)

    # If we don't have enough species champions, fill from population
    if len(champions) < n:
        champions = list(pop.population.values())

    # Sort by fitness and return top n
    champions.sort(key=lambda g: g.fitness or 0, reverse=True)
    return champions[:n]

def evaluate_against_parasites(host_pop, parasite_pop, hall_of_fame, config, desc="Evaluating"):
    '''
    Host population is evaluated against a sample of parasites from the parasite population:
    - 4 highest species champions
    - 8 random selected genomes from an hall of fame of all generations' champions
    '''
    species_champions = get_species_champions(parasite_pop, n=4)
    hof_sample = random.sample(hall_of_fame, min(8, len(hall_of_fame))) if hall_of_fame else []
    parasite_sample = species_champions + hof_sample

    parasite_nets = [neat.nn.RecurrentNetwork.create(g, config) for g in parasite_sample]

    game = Game(render=False)
    for host_genome_id, host_genome in tqdm(host_pop.population.items(), desc=desc, leave=False):
        host_net = neat.nn.RecurrentNetwork.create(host_genome, config)
        host_genome.fitness = 0
        for parasite_net in parasite_nets:
            f1, _ = match(game, host_net, parasite_net)
            host_genome.fitness += f1
            f1, _ = match(game, host_net, parasite_net, inv=True)
            host_genome.fitness += f1

    champion_id, champion_genome = max(host_pop.population.items(), key=lambda item: item[1].fitness or 0)
    hall_of_fame.append(copy.deepcopy(champion_genome))
    return champion_genome

EXTRA_ENERGY_POINTS_RIGHT = [
    (350, 100), (450, 100), (550, 100),
    (350, 233), (450, 233), (550, 233),
    (350, 366), (450, 366), (550, 366),
    (350, 500), (450, 500), (550, 500)
]
EXTRA_ENERGY_POINTS_LEFT = [
    (250, 100), (150, 100), (50, 100),
    (250, 233), (150, 233), (50, 233),
    (250, 366), (150, 366), (50, 366),
    (250, 500), (150, 500), (50, 500)
]

def get_gen_champion(champion1, champion2, config):
    game = Game(render=False)
    net1 = neat.nn.RecurrentNetwork.create(champion1, config)
    net2 = neat.nn.RecurrentNetwork.create(champion2, config)
    fitness1 = 0
    fitness2 = 0
    for extra_left in EXTRA_ENERGY_POINTS_LEFT:
        for extra_right in EXTRA_ENERGY_POINTS_RIGHT:
            f1, f2 = match(game, net1, net2, extra_energy_1=extra_left, extra_energy_2=extra_right)
            fitness1 += f1
            fitness2 += f2
            f1, f2 = match(game, net1, net2, inv=True, extra_energy_1=extra_left, extra_energy_2=extra_right)
            fitness1 += f1
            fitness2 += f2

    if fitness1 >= fitness2:
        return champion1
    else:
        return champion2

def dominance_tournament(generation_champion, dominant_strategies, config):
    """
    Test if generation_champion beats ALL previous dominant strategies.
    If so, add it to the dominant strategies list.
    """
    # First generation champion is automatically the first dominant strategy
    if not dominant_strategies:
        dominant_strategies.append(copy.deepcopy(generation_champion))
        return

    game = Game(render=False)
    net_champion = neat.nn.RecurrentNetwork.create(generation_champion, config)

    for dominant_strategy in dominant_strategies:
        net_dominant = neat.nn.RecurrentNetwork.create(dominant_strategy, config)
        fitness_champion, fitness_dominant = 0, 0

        for extra_left in EXTRA_ENERGY_POINTS_LEFT:
            for extra_right in EXTRA_ENERGY_POINTS_RIGHT:
                f1, f2 = match(game, net_champion, net_dominant, extra_energy_1=extra_left, extra_energy_2=extra_right)
                fitness_champion += f1
                fitness_dominant += f2
                f1, f2 = match(game, net_champion, net_dominant, inv=True, extra_energy_1=extra_left, extra_energy_2=extra_right)
                fitness_champion += f1
                fitness_dominant += f2

        # Must beat (not just tie) each dominant strategy
        if fitness_champion <= fitness_dominant:
            return  # Failed to beat this one, not a new dominant strategy

    # Beat all previous dominant strategies
    dominant_strategies.append(copy.deepcopy(generation_champion))

def save_checkpoint(filepath, pop1, pop2, hall_of_fame1, hall_of_fame2, dominant_strategies, generation):
    """Save training state to a checkpoint file."""
    checkpoint = {
        'pop1': pop1,
        'pop2': pop2,
        'hall_of_fame1': hall_of_fame1,
        'hall_of_fame2': hall_of_fame2,
        'dominant_strategies': dominant_strategies,
        'generation': generation
    }
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)

def load_checkpoint(filepath):
    """Load training state from a checkpoint file."""
    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)
    return (
        checkpoint['pop1'],
        checkpoint['pop2'],
        checkpoint['hall_of_fame1'],
        checkpoint['hall_of_fame2'],
        checkpoint['dominant_strategies'],
        checkpoint['generation']
    )

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.txt', help='Path to NEAT config file')
    parser.add_argument('--generations', type=int, default=500, help='Number of generations to train')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file to resume from')
    parser.add_argument('--checkpoint-interval', type=int, default=10, help='Save checkpoint every N generations')
    args = parser.parse_args()

    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)

    # Load configuration
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        args.config
    )

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        pop1, pop2, hall_of_fame1, hall_of_fame2, dominant_strategies, start_generation = load_checkpoint(args.resume)
        start_generation += 1  # Start from the next generation
    else:
        pop1 = neat.Population(config)
        pop2 = neat.Population(config)
        dominant_strategies = []
        hall_of_fame1 = []
        hall_of_fame2 = []
        start_generation = 0

    try:
        for generation in tqdm(range(start_generation, args.generations), desc="Generations", position=0):
            # Evaluate pop1 using pop2 as parasites
            champion1 = evaluate_against_parasites(pop1, pop2, hall_of_fame2, config, desc="Pop1 eval")

            # Evaluate pop2 using pop1 as parasites
            champion2 = evaluate_against_parasites(pop2, pop1, hall_of_fame1, config, desc="Pop2 eval")

            generation_champion = get_gen_champion(champion1, champion2, config)

            dominance_tournament(generation_champion, dominant_strategies, config)

            # Reproduce both populations
            pop1.species.speciate(config, pop1.population, generation)
            pop1.population = pop1.reproduction.reproduce(config, pop1.species, config.pop_size, generation)
            pop1.species.speciate(config, pop1.population, generation)  # Re-speciate new population

            pop2.species.speciate(config, pop2.population, generation)
            pop2.population = pop2.reproduction.reproduce(config, pop2.species, config.pop_size, generation)
            pop2.species.speciate(config, pop2.population, generation)  # Re-speciate new population

            # Save checkpoint periodically
            if (generation + 1) % args.checkpoint_interval == 0:
                checkpoint_path = f'checkpoints/checkpoint-gen-{generation + 1}.pkl'
                save_checkpoint(checkpoint_path, pop1, pop2, hall_of_fame1, hall_of_fame2, dominant_strategies, generation)
                tqdm.write(f"Saved checkpoint: {checkpoint_path} | Dominant strategies: {len(dominant_strategies)}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # Save final state
        print(f"\nSaving final state...")
        save_checkpoint('checkpoints/checkpoint-final.pkl', pop1, pop2, hall_of_fame1, hall_of_fame2, dominant_strategies, generation)
