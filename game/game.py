import pygame
import random
from .energy import EnergyPoint
from .robot import Robot, SCREEN_WIDTH, SCREEN_HEIGHT

class Game:
    def __init__(self, render=True):
        self.width = SCREEN_WIDTH
        self.height = SCREEN_HEIGHT
        self.render = render
        self.winner = None

        if self.render:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Robot War")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)

        self.reset()

    def reset(self):
        '''  
        Reset game to initial deterministic position.
        '''
        self.energy_points = [
            EnergyPoint((300, 75), self.render),
            EnergyPoint((150, 200), self.render),
            EnergyPoint((450, 200), self.render),
            EnergyPoint((50, 300), self.render),
            EnergyPoint((550, 300), self.render),
            EnergyPoint((150, 400), self.render),
            EnergyPoint((450, 400), self.render),
            EnergyPoint((250, 500), self.render),
            EnergyPoint((350, 500), self.render)
            ]
        self.robot1 = Robot(x=200, y=300, angle=180, energy_points=self.energy_points, color=(0, 0, 255), render=self.render)
        self.robot2 = Robot(x=400, y=300, angle=0, energy_points=self.energy_points, color=(255, 0, 0), render=self.render)
        self.game_over = False
        self.winner = None
        self.collision_kill = False  # True if game ended by collision
        self.frame_count = 0
        self.max_frames = 750  # Max game duration

    def check_game_over(self):
        self.frame_count += 1
        # Check for collision first
        if self.robot1.alive == False or self.robot2.alive == False:
            self.game_over = True
            self.collision_kill = True
            # In a collision, both robots die - winner has more energy
            if self.robot1.energy > self.robot2.energy:
                self.winner = 1
            elif self.robot2.energy > self.robot1.energy:
                self.winner = 2
            else:
                self.winner = 0  # Draw (same energy)
        # Check for timeout
        elif self.frame_count >= self.max_frames:
            self.game_over = True
            self.collision_kill = False
            if self.robot1.energy > self.robot2.energy:
                self.winner = 1
            elif self.robot2.energy > self.robot1.energy:
                self.winner = 2
            else:
                self.winner = 0  # Draw (same energy)
    
    def play(self, action1, action2):
        '''
        action1, action2: tuples of (left, right, forward) continuous floats in [0,1]
        '''
        left1, right1, forward1 = action1
        left2, right2, forward2 = action2

        if self.render:
            self.robot1.draw(self.screen)
            self.robot1.update(left1, right1, forward1, self.robot2)
            self.robot2.draw(self.screen)
            self.robot2.update(left2, right2, forward2, self.robot1)
            pygame.display.update()
        else:
            self.robot1.update(left1, right1, forward1, self.robot2)
            self.robot2.update(left2, right2, forward2, self.robot1)

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
            action1 = (random.random(), random.random(), random.random())
            action2 = (random.random(), random.random(), random.random())
            game.play(action1, action2)
            game.check_game_over()
