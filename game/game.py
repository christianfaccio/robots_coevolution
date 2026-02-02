import pygame
import random
from .energy import EnergyPoint
from .robot import Robot, SCREEN_WIDTH, SCREEN_HEIGHT, directions

class Game:
    def __init__(self, render=True):
        self.width = SCREEN_WIDTH
        self.height = SCREEN_HEIGHT
        self.render = render
        self.winner = None

        if self.render:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Robot Survivor")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)

        self.reset()

    def reset(self):
        '''  
        Reset game to initial deterministic position.
        '''
        self.energy_points = [
            EnergyPoint((100, 50), self.render),
            EnergyPoint((275, 497), self.render),
            EnergyPoint((571, 300), self.render),
            EnergyPoint((400, 235), self.render),
            EnergyPoint((800, 600), self.render),
            EnergyPoint((1000, 800), self.render),
            EnergyPoint((600, 900), self.render)
            ]
        self.robot1 = Robot(x=200, y=200, energy_points=self.energy_points, color=(0, 0, 255), render=self.render)
        self.robot2 = Robot(x=500, y=500, energy_points=self.energy_points, color=(255, 0, 0), render=self.render)
        self.game_over = False
        self.winner = None
        self.frame_count = 0
        self.max_frames = 2000  # Max game duration

    def check_game_over(self):
        self.frame_count += 1
        # Collision or timeout - winner is always determined by energy
        if (self.robot1.alive == False or self.robot2.alive == False or
            self.frame_count >= self.max_frames):
            self.game_over = True
            if self.robot1.energy > self.robot2.energy:
                self.winner = 1
            elif self.robot2.energy > self.robot1.energy:
                self.winner = 2
            else:
                self.winner = 0  # Draw (same energy)
    
    def play(self, action1=None, action2=None):
        directions_list = list(directions)
        self.robot1.direction = directions_list[action1]
        self.robot2.direction = directions_list[action2]

        if self.render:
            self.robot1.draw(self.screen)
            self.robot1.update(self.robot2, self.screen)
            self.robot2.draw(self.screen)
            self.robot2.update(self.robot1, self.screen)
            pygame.display.update()
        else:
            self.robot1.update(self.robot2)
            self.robot2.update(self.robot1)

if __name__ == '__main__':
    game = Game(render=True)
    running = True
    while running: 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        game.screen.fill((0, 0, 0))  # Clear screen with black
        for ep in game.energy_points:
            ep.draw(game.screen)
        game.robot1.draw(game.screen)
        game.robot2.draw(game.screen)
        pygame.display.flip()
        game.clock.tick(30)

        if not game.game_over:
            action1 = random.randint(0,2)
            action2 = random.randint(0,2)
            game.play(action1, action2)
            game.check_game_over()
