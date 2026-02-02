import pygame
import enum
import math

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 1000
ROBOT_SIZE = 40

directions = enum.Enum('directions', 'LEFT STRAIGHT RIGHT')

class Robot:
    def __init__(self, x, y, energy_points, color=(0, 0, 255), render=True):
        self.render = render
        self.color = color
        self.vel_vector = pygame.math.Vector2(0.8, 0)
        self.speed_factor = 6
        self.angle = 0
        self.rotation_vel = 5
        self.direction = directions.STRAIGHT
        self.alive = True
        self.radars = []
        self.energy_points = energy_points
        self.energy = 0

        self.rect = pygame.Rect(0, 0, ROBOT_SIZE, ROBOT_SIZE)
        self.rect.center = (x, y)


    def update(self, other_robot, screen=None):
        self.radars.clear()
        self.translate()
        self.rotate()
        self.radar(other_robot, screen)
        self.collision(other_robot)

    def translate(self):
        self.rect.center += self.vel_vector * self.speed_factor
        # if robot is out of screen, keep it inside
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > SCREEN_WIDTH:
            self.rect.right = SCREEN_WIDTH
        if self.rect.top < 0:
            self.rect.top = 0
        if self.rect.bottom > SCREEN_HEIGHT:
            self.rect.bottom = SCREEN_HEIGHT

    def rotate(self):
        if self.direction == directions.LEFT:
            self.angle += self.rotation_vel
            self.vel_vector.rotate_ip(self.rotation_vel)
        elif self.direction == directions.RIGHT:
            self.angle -= self.rotation_vel
            self.vel_vector.rotate_ip(-self.rotation_vel)

    def collision(self, other_robot):
        '''
        The robot dies if it collides with another robot.
        '''
        if self.rect.colliderect(other_robot.rect):
            self.alive = False
        # Check collision with energy points
        for energy_point in self.energy_points:
            if self.rect.colliderect(energy_point.rect):
                self.energy += 100
                self.energy_points.remove(energy_point)
    
    def radar(self, other_robot, screen=None):
        '''
        Measure distance to walls, energy points and enemy robot in the direction of the radars.
        '''
        length = 0
        x = int(self.rect.center[0])
        y = int(self.rect.center[1])

        # Distance to walls
        self.radars.append(SCREEN_WIDTH - x)  # right wall
        self.radars.append(x)                 # left wall
        self.radars.append(SCREEN_HEIGHT - y) # bottom wall
        self.radars.append(y)                 # top wall

        if self.render and screen:
            pygame.draw.line(screen, (255, 0, 0), (x, y), (SCREEN_WIDTH, y), 1)  # right
            pygame.draw.circle(screen, (255, 0, 0), (x, y), 4)
            pygame.draw.line(screen, (255, 0, 0), (x, y), (0, y), 1)             # left
            pygame.draw.circle(screen, (255, 0, 0), (x, y), 4)
            pygame.draw.line(screen, (255, 0, 0), (x, y), (x, SCREEN_HEIGHT), 1) # bottom
            pygame.draw.circle(screen, (255, 0, 0), (x, y), 4)
            pygame.draw.line(screen, (255, 0, 0), (x, y), (x, 0), 1)              # top
            pygame.draw.circle(screen, (255, 0, 0), (x, y), 4)

        # Distance to closest energy point
        closest = None
        for energy_point in self.energy_points:
            dist = math.hypot(energy_point.rect.center[0] - x, energy_point.rect.center[1] - y)
            if length == 0 or dist < length:
                length = dist
                closest = energy_point
        self.radars.append(int(length))

        if self.render and screen and closest is not None:
            pygame.draw.line(screen, (0, 255, 0), (x, y), (closest.rect.center[0], closest.rect.center[1]), 1)
            pygame.draw.circle(screen, (0, 255, 0), (closest.rect.center[0], closest.rect.center[1]), 4)

        # Distance to enemy robot
        dist = math.hypot(other_robot.rect.center[0] - x, other_robot.rect.center[1] - y)
        self.radars.append(int(dist))

        if self.render and screen:
            pygame.draw.line(screen, (0, 0, 255), (x, y), (other_robot.rect.center[0], other_robot.rect.center[1]), 1)
            pygame.draw.circle(screen, (0, 0, 255), (other_robot.rect.center[0], other_robot.rect.center[1]), 4)
    
    def state(self, other_robot):
        self.radars.clear()
        self.radar(other_robot)
        # walls (4), closest energy point (1), enemy robot (1), my energy (1), enemy energy (1)
        inputs = []
        for radar in self.radars:
            inputs.append(radar)
        inputs.append(self.energy)
        inputs.append(other_robot.energy)
        return inputs
    
    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
