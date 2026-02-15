import pickle
import random
import argparse
import neat
import pygame
from game.game import Game


def load_genome_from_checkpoint(checkpoint_path):
    """Load the best genome from a training checkpoint."""
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)

    # Get the last dominant strategy (best genome)
    if checkpoint.get('dominant_strategies'):
        return checkpoint['dominant_strategies'][-1]

    # Fallback: get best from hall of fame
    hof1 = checkpoint.get('hall_of_fame1', [])
    hof2 = checkpoint.get('hall_of_fame2', [])
    all_genomes = hof1 + hof2

    if all_genomes:
        return max(all_genomes, key=lambda g: g.fitness or 0)

    raise ValueError("No genomes found in checkpoint")


def load_genome(path):
    """Load a genome from either a checkpoint or a direct genome pickle."""
    with open(path, 'rb') as f:
        data = pickle.load(f)

    # Check if it's a checkpoint (has 'dominant_strategies' or 'hall_of_fame1')
    if isinstance(data, dict) and ('dominant_strategies' in data or 'hall_of_fame1' in data):
        return load_genome_from_checkpoint(path)

    # Otherwise assume it's a direct genome
    return data


def play(config_path, genome_path='random', opponent='random'):
    """
    Play a visual game with a trained genome.

    Args:
        config_path: Path to NEAT config file
        genome_path: random or path to genome pickle or checkpoint
        opponent: 'random' or path to another genome/checkpoint
    """
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    # Load trained genome
    net = None 
    if genome_path != 'random':
        genome = load_genome(genome_path)
        net = neat.nn.RecurrentNetwork.create(genome, config)

    # Load opponent if not random
    opponent_net = None
    if opponent != 'random':
        opponent_genome = load_genome(opponent)
        opponent_net = neat.nn.RecurrentNetwork.create(opponent_genome, config)

    game = Game(render=True)
    hud_font = pygame.font.Font(None, 22)

    running = True
    wins = {'trained': 0, 'opponent': 0, 'draw': 0}
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not game.game_over:
            game.screen.fill((0, 0, 0))

            for ep in game.energy_points:
                ep.draw(game.screen)

            # HUD - Robot 1 (blue, left side)
            r1 = game.robot1
            for i, line in enumerate([
                f"Food: {r1.food_collected}",
                f"Dist: {r1.distance_traveled:.0f}px",
                f"Energy: {r1.energy:.0f}",
            ]):
                surf = hud_font.render(line, True, (100, 100, 255))
                game.screen.blit(surf, (10, 10 + i * 20))

            # HUD - Robot 2 (red, right side)
            r2 = game.robot2
            for i, line in enumerate([
                f"Food: {r2.food_collected}",
                f"Dist: {r2.distance_traveled:.0f}px",
                f"Energy: {r2.energy:.0f}",
            ]):
                surf = hud_font.render(line, True, (255, 100, 100))
                w = surf.get_width()
                game.screen.blit(surf, (game.width - w - 10, 10 + i * 20))

            # Trained robot (blue, robot1)
            if net:
                state1 = game.robot1.state(game.robot2)
                output1 = net.activate(state1)
            else:
                state1 = game.robot1.state(game.robot2)
                output1 = (random.random(), random.random(), random.random())

            # Print robot1 inputs/outputs
            print(f"IN:  {' | '.join(f'{v:.2f}' for v in state1)}")
            print(f"OUT: L={output1[0]:.2f} R={output1[1]:.2f} F={output1[2]:.2f}")

            # Opponent (red, robot2)
            if opponent_net:
                state2 = game.robot2.state(game.robot1)
                output2 = opponent_net.activate(state2)
            else:
                output2 = (random.random(), random.random(), random.random())

            game.play(output1, output2)

            game.clock.tick(30)
        else:
            # Update win counter
            if game.winner == 1:
                wins['trained'] += 1
            elif game.winner == 2:
                wins['opponent'] += 1
            else:
                wins['draw'] += 1

            font = pygame.font.Font(None, 74)
            small_font = pygame.font.Font(None, 36)

            if game.winner == 1:
                text = font.render("Robot 1 Wins!", True, (0, 255, 0))
            elif game.winner == 2:
                text = font.render("Robot 2 Wins!", True, (255, 0, 0))
            else:
                text = font.render("Draw!", True, (255, 255, 0))

            text_rect = text.get_rect(center=(game.width // 2, game.height // 2))
            game.screen.blit(text, text_rect)

            hint_text = small_font.render("SPACE to play again | ESC to quit", True, (150, 150, 150))
            hint_rect = hint_text.get_rect(center=(game.width // 2, game.height // 2 + 100))
            game.screen.blit(hint_text, hint_rect)

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        game.reset()
                        if net:
                            net.reset()
                        if opponent_net:
                            opponent_net.reset()
                    elif event.key == pygame.K_ESCAPE:
                        running = False

    pygame.quit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play a game with a trained genome')
    parser.add_argument('--config', type=str, default='config.txt',
                        help='Path to NEAT config')
    parser.add_argument('--genome', type=str, default='random',
                        help='Path to trained genome (pickle or checkpoint) or "random" for a random genome')
    parser.add_argument('--opponent', type=str, default='random',
                        help='Opponent: "random" or path to another genome/checkpoint')
    args = parser.parse_args()

    play(args.config, args.genome, args.opponent)
