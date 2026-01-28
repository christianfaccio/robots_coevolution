import pygame
import math
import random
from .bullet import Bullet


class Robot(pygame.sprite.Sprite):
    def __init__(self, x, y, color):
        super().__init__()
        self.x = x
        self.y = y
        self.angle = random.uniform(-1,1) * math.pi  
        self.shoot_angle = random.uniform(-1,1) * math.pi  
        self.color = color
        self.id = id(self)  # Unique identifier

        # Physical properties
        self.radius = 10
        self.speed = 0
        self.max_speed = 1

        # Combat properties
        self.health = 100
        self.alive = True
        self.shoot_cooldown = 0
        self.cooldown_time = 30  # Frames between shots

        # Sensor configuration: circle with range
        self.sensor_range = 150

        # Create surface for the robot
        self.image = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(self.image, self.color, (self.radius, self.radius), self.radius)
        self.rect = self.image.get_rect(center=(int(self.x), int(self.y)))

    def update(self, arena_width, arena_height, action=None, obstacles=None):
        """Update robot state based on action.

        action: dict with keys:
            - 'movement': tuple float for robot rotation (fixed speed)
            - 'turret': float for turret rotation
            - 'shoot': bool for shooting
        obstacles: list of Obstacle objects to check collision against
        """
        if not self.alive:
            return None

        bullet = None

        if action:
            # Movement
            if 'movement' in action:
                self.speed = self.max_speed
                self.angle += action['movement']

            # Rotation
            if 'turret' in action:
                self.shoot_angle += action['turret']

            # Shooting
            if action.get('shoot', False) and self.shoot_cooldown <= 0:
                bullet = self.shoot()
                self.shoot_cooldown = self.cooldown_time

        # Store previous position for obstacle collision
        prev_x = self.x
        prev_y = self.y

        # Apply movement
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed

        # Boundary collision
        self.x = max(self.radius, min(arena_width - self.radius, self.x))
        self.y = max(self.radius, min(arena_height - self.radius, self.y))

        # Obstacle collision - revert to previous position if colliding
        if obstacles:
            for obstacle in obstacles:
                if obstacle.collides_with_circle(self.x, self.y, self.radius):
                    self.x = prev_x
                    self.y = prev_y
                    break

        # Update rect position
        self.rect.center = (int(self.x), int(self.y))

        # Decrease shoot cooldown
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1

        return bullet

    def shoot(self):
        """Create a bullet traveling in the robot's facing direction."""
        # Spawn bullet slightly in front of the robot
        bullet_x = self.x
        bullet_y = self.y
        return Bullet(bullet_x, bullet_y, self.shoot_angle, self.id)

    def take_damage(self, damage):
        """Reduce health when hit."""
        self.health -= damage
        if self.health <= 0:
            self.health = 0
            self.alive = False

    def update_sensors(self, other_robot, arena_width, arena_height, screen=None, obstacles=None):
        """Update distance sensor readings.

        Sensors detect:
        - Distance to walls
        - Distance to enemy robot (if in sensor cone)
        - Distance to obstacles
        """
        self.walls_distances = [
            (arena_height - self.y) / self.sensor_range if (arena_height - self.y) < self.sensor_range else 1.0, # South
            (arena_width - self.x) / self.sensor_range if (arena_width - self.x) < self.sensor_range else 1.0, # East
            self.y / self.sensor_range if self.y < self.sensor_range else 1.0, # North
            self.x / self.sensor_range if self.x < self.sensor_range else 1.0 # West
        ] 
        self.enemy_distance = math.hypot(other_robot.x - self.x, other_robot.y - self.y) / self.sensor_range if math.hypot(other_robot.x - self.x, other_robot.y - self.y) < self.sensor_range else 1.0
        self.obstacles_distances = [(obstacle.id, math.hypot(obstacle.x - self.x, obstacle.y - self.y) / self.sensor_range) for obstacle in obstacles if math.hypot(obstacle.x - self.x, obstacle.y - self.y) < self.sensor_range] if obstacles else []

        # Draw sensor circle range if screen provided
        if screen:
            pygame.draw.circle(screen, (0, 255, 0), (int(self.x), int(self.y)), self.sensor_range, 1)

        return self.walls_distances + [self.enemy_distance] + self.obstacles_distances

    def get_state(self, other_robot, arena_width, arena_height):
        """Get the robot's state as input for AI/neural network.

        Returns a dictionary.
        """
        # TODO: check for other_robot being alive
        state = {
            'x': self.x / arena_width,
            'y': self.y / arena_height,
            'angle': self.angle / math.pi,
            'enemy_distance': self.enemy_distance,
            'angle_to_enemy': (math.atan2(other_robot.y - self.y, other_robot.x) - self.angle) / math.pi,
            'enemy_health': other_robot.health / 100,
            'self_health': self.health / 100,
            'time_to_shoot': self.shoot_cooldown / self.cooldown_time,
            'obstacles_distances': self.obstacles_distances,
            'walls_distances': self.walls_distances
        }
        return state

    def draw(self, screen):
        """Draw the robot on screen."""
        if not self.alive:
            # Draw dead robot as gray
            pygame.draw.circle(screen, (100, 100, 100), (int(self.x), int(self.y)), self.radius)
            return

        # Draw robot body
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

        # Draw health bar above robot
        bar_width = self.radius * 4
        bar_height = self.radius / 2
        bar_x = self.x - bar_width / 2
        bar_y = self.y - self.radius - 10

        # Background (red)
        pygame.draw.rect(screen, (255, 0, 0),
                        (int(bar_x), int(bar_y), bar_width, bar_height))
        # Health (green)
        health_width = bar_width * (self.health / 100)
        pygame.draw.rect(screen, (0, 255, 0),
                        (int(bar_x), int(bar_y), int(health_width), bar_height))
