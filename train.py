import os
import copy
import time
import neat
import random
import pickle
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
from game.game import Game

# Number of parallel workers (use all available cores)
NUM_WORKERS = mp.cpu_count()

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

def evaluate_genome_worker(args):
    """Worker function to evaluate a single genome against all parasites."""
    genome_id, genome, parasite_genomes, config = args

    # Create networks (must be done in worker process)
    host_net = neat.nn.RecurrentNetwork.create(genome, config)
    parasite_nets = [neat.nn.RecurrentNetwork.create(g, config) for g in parasite_genomes]

    game = Game(render=False)
    fitness = 0
    for parasite_net in parasite_nets:
        f1, _ = match(game, host_net, parasite_net)
        fitness += f1
        f1, _ = match(game, host_net, parasite_net, inv=True)
        fitness += f1

    return genome_id, fitness

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

def evaluate_against_parasites(host_pop, parasite_pop, hall_of_fame, config, pool, desc="Evaluating"):
    '''
    Host population is evaluated against a sample of parasites from the parasite population:
    - 4 highest species champions
    - 8 random selected genomes from an hall of fame of all generations' champions
    '''
    species_champions = get_species_champions(parasite_pop, n=4)
    hof_sample = random.sample(hall_of_fame, min(8, len(hall_of_fame))) if hall_of_fame else []
    parasite_genomes = species_champions + hof_sample

    # Prepare args for parallel evaluation
    eval_args = [
        (genome_id, genome, parasite_genomes, config)
        for genome_id, genome in host_pop.population.items()
    ]

    # Parallel evaluation with progress bar
    results = list(tqdm(
        pool.imap_unordered(evaluate_genome_worker, eval_args),
        total=len(eval_args),
        desc=desc,
        leave=False
    ))

    # Update fitness values
    for genome_id, fitness in results:
        host_pop.population[genome_id].fitness = fitness

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

def champion_match_worker(args):
    """Worker function for champion comparison matches."""
    genome1, genome2, config, extra_left, extra_right = args

    net1 = neat.nn.RecurrentNetwork.create(genome1, config)
    net2 = neat.nn.RecurrentNetwork.create(genome2, config)
    game = Game(render=False)

    f1_total, f2_total = 0, 0

    f1, f2 = match(game, net1, net2, extra_energy_1=extra_left, extra_energy_2=extra_right)
    f1_total += f1
    f2_total += f2

    f1, f2 = match(game, net1, net2, inv=True, extra_energy_1=extra_left, extra_energy_2=extra_right)
    f1_total += f1
    f2_total += f2

    return f1_total, f2_total

def get_gen_champion(champion1, champion2, config, pool):
    # Prepare all match configurations
    match_args = [
        (champion1, champion2, config, extra_left, extra_right)
        for extra_left in EXTRA_ENERGY_POINTS_LEFT
        for extra_right in EXTRA_ENERGY_POINTS_RIGHT
    ]

    # Run matches in parallel
    results = pool.map(champion_match_worker, match_args)

    fitness1 = sum(r[0] for r in results)
    fitness2 = sum(r[1] for r in results)

    if fitness1 >= fitness2:
        return champion1
    else:
        return champion2

def dominance_match_worker(args):
    """Worker function for dominance tournament matches."""
    champion, dominant, config, extra_left, extra_right = args

    net_champion = neat.nn.RecurrentNetwork.create(champion, config)
    net_dominant = neat.nn.RecurrentNetwork.create(dominant, config)
    game = Game(render=False)

    f_champ, f_dom = 0, 0

    f1, f2 = match(game, net_champion, net_dominant, extra_energy_1=extra_left, extra_energy_2=extra_right)
    f_champ += f1
    f_dom += f2

    f1, f2 = match(game, net_champion, net_dominant, inv=True, extra_energy_1=extra_left, extra_energy_2=extra_right)
    f_champ += f1
    f_dom += f2

    return f_champ, f_dom

def dominance_tournament(generation_champion, dominant_strategies, config, pool):
    """
    Test if generation_champion beats ALL previous dominant strategies.
    If so, add it to the dominant strategies list.
    Returns True if a new dominant strategy was added.
    """
    # First generation champion is automatically the first dominant strategy
    if not dominant_strategies:
        dominant_strategies.append(copy.deepcopy(generation_champion))
        tqdm.write(f"*** NEW DOMINANT STRATEGY #{len(dominant_strategies)} (first) ***")
        return True

    for i, dominant_strategy in enumerate(dominant_strategies):
        # Prepare all match configurations for this opponent
        match_args = [
            (generation_champion, dominant_strategy, config, extra_left, extra_right)
            for extra_left in EXTRA_ENERGY_POINTS_LEFT
            for extra_right in EXTRA_ENERGY_POINTS_RIGHT
        ]

        # Run matches in parallel
        results = pool.map(dominance_match_worker, match_args)

        fitness_champion = sum(r[0] for r in results)
        fitness_dominant = sum(r[1] for r in results)

        # Must beat (not just tie) each dominant strategy
        if fitness_champion <= fitness_dominant:
            return False  # Failed to beat this one, not a new dominant strategy
        
        tqdm.write(f"Generation champion beats dominant strategy #{i+1} with score {fitness_champion} vs {fitness_dominant}")

    # Beat all previous dominant strategies
    dominant_strategies.append(copy.deepcopy(generation_champion))
    tqdm.write(f"*** NEW DOMINANT STRATEGY #{len(dominant_strategies)} ***")
    return True

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
    parser.add_argument('--workers', type=int, default=NUM_WORKERS, help='Number of parallel workers')
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

    print(f"Using {args.workers} parallel workers")

    # Create process pool
    with mp.Pool(processes=args.workers) as pool:
        try:
            for generation in tqdm(range(start_generation, args.generations), desc="Generations", position=0):
                t0 = time.time()

                # Evaluate pop1 using pop2 as parasites
                champion1 = evaluate_against_parasites(pop1, pop2, hall_of_fame2, config, pool, desc="Pop1 eval")
                t1 = time.time()

                # Evaluate pop2 using pop1 as parasites
                champion2 = evaluate_against_parasites(pop2, pop1, hall_of_fame1, config, pool, desc="Pop2 eval")
                t2 = time.time()

                generation_champion = get_gen_champion(champion1, champion2, config, pool)
                t3 = time.time()

                new_dominant = dominance_tournament(generation_champion, dominant_strategies, config, pool)
                t4 = time.time()

                # Reproduce both populations
                pop1.species.speciate(config, pop1.population, generation)
                pop1.population = pop1.reproduction.reproduce(config, pop1.species, config.pop_size, generation)
                pop1.species.speciate(config, pop1.population, generation)  # Re-speciate new population

                pop2.species.speciate(config, pop2.population, generation)
                pop2.population = pop2.reproduction.reproduce(config, pop2.species, config.pop_size, generation)
                pop2.species.speciate(config, pop2.population, generation)  # Re-speciate new population
                t5 = time.time()

                tqdm.write(f"Gen {generation} | Pop1: {t1-t0:.1f}s | Pop2: {t2-t1:.1f}s | "
                           f"Champion: {t3-t2:.1f}s | Dominance: {t4-t3:.1f}s | Repro: {t5-t4:.1f}s | "
                           f"Total: {t5-t0:.1f}s")

                # Save checkpoint when a new dominant strategy is found
                if new_dominant:
                    checkpoint_path = f'checkpoints/checkpoint-dominant-{len(dominant_strategies)}.pkl'
                    save_checkpoint(checkpoint_path, pop1, pop2, hall_of_fame1, hall_of_fame2, dominant_strategies, generation)
                    tqdm.write(f"Saved checkpoint: {checkpoint_path} | Dominant strategies: {len(dominant_strategies)}")

        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
        finally: 
            # Save final checkpoint
            checkpoint_path = f'checkpoints/final_checkpoint.pkl'
            save_checkpoint(checkpoint_path, pop1, pop2, hall_of_fame1, hall_of_fame2, dominant_strategies, generation)
            print(f"Saved final checkpoint: {checkpoint_path} | Dominant strategies: {len(dominant_strategies)}")