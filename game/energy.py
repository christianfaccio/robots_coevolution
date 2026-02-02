import pygame

ENERGY_SIZE = 15

class EnergyPoint:
    def __init__(self, position, render=True):
        self.render = render
        self.color = (255, 255, 0)  # Yellow
        self.rect = pygame.Rect(0, 0, ENERGY_SIZE, ENERGY_SIZE)
        self.rect.center = position

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)

    