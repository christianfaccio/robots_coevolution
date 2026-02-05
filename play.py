import pickle
import random
import neat
import pygame
from game.game import Game


def play():
    """
    For now a random play.
    """
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        'config.txt'
    )

    game = Game(render=True)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not game.game_over:
            game.screen.fill((0, 0, 0))

            for ep in game.energy_points:
                ep.draw(game.screen)

            output1 = (random.random(), random.random(), random.random()) # random.random() has range [0,1]
            output2 = (random.random(), random.random(), random.random())

            game.play(output1, output2)
            game.check_game_over()
            game.clock.tick(30)
        else:
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
    play()
