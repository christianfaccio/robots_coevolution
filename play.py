import pickle
import neat
import pygame
from game.game import Game

def extract_best_genome(checkpoint_path):
      """Extract the best genome from a NEAT checkpoint file."""
      # Restore the checkpoint
      pop = neat.Checkpointer.restore_checkpoint(checkpoint_path)

      # Find the best genome across all species
      best_genome = None
      best_fitness = float('-inf')

      for species_id, species in pop.species.species.items():
          for genome_id, genome in species.members.items():
              if genome.fitness is not None and genome.fitness > best_fitness:
                  best_fitness = genome.fitness
                  best_genome = genome

      return best_genome

def play(config_path, checkpoint_path, opponent_checkpoint_path):
    """
    Run a visual simulation with a trained genome.

    Args:
        config_path: Path to NEAT config file
        genome_path: Path to saved genome pickle file
        opponent: 'random' for random actions, or path to another genome
    """
    # Load NEAT config
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    genome = extract_best_genome(checkpoint_path)
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Load opponent network if provided
    opponent_net = None
    if opponent_checkpoint_path != 'random':
        opponent_genome = extract_best_genome(opponent_checkpoint_path)
        opponent_net = neat.nn.FeedForwardNetwork.create(opponent_genome, config)

    # Create game with rendering
    game = Game(render=True)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not game.game_over:
            game.screen.fill((0, 0, 0))

            # Draw energy points
            for ep in game.energy_points:
                ep.draw(game.screen)

            # Get action for trained robot
            state1 = game.robot1.state(game.robot2)
            output1 = net.activate(state1)
            action1 = output1.index(max(output1))

            # Get action for opponent
            if opponent_net:
                state2 = game.robot2.state(game.robot1)
                output2 = opponent_net.activate(state2)
                action2 = output2.index(max(output2))
            else:
                import random
                action2 = random.randint(0, 2)

            game.play(action1, action2)
            game.check_game_over()
            game.clock.tick(30)
        else:
            # Display winner and wait
            font = pygame.font.Font(None, 74)
            if game.winner == 1:
                text = font.render("Robot 1 Wins!", True, (0, 255, 0))
            elif game.winner == 2:
                text = font.render("Robot 2 Wins!", True, (255, 0, 0))
            else:
                text = font.render("Draw!", True, (255, 255, 0))

            text_rect = text.get_rect(center=(game.width // 2, game.height // 2))
            game.screen.blit(text, text_rect)
            pygame.display.flip()

            # Wait for quit or restart
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        game.reset()
                    elif event.key == pygame.K_ESCAPE:
                        running = False

    pygame.quit()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Play a game with a trained genome')
    parser.add_argument('--config', type=str, default='config.txt', help='Path to NEAT config')
    parser.add_argument('--checkpoint', type=str, default='best_genome.pkl', help='Path to trained genome checkpoint')
    parser.add_argument('--opponent', type=str, default='random',
                        help='Opponent: "random" or path to another genome checkpoint')
    args = parser.parse_args()

    play(args.config, args.checkpoint, args.opponent)
