import pygame
import math
from .bullet import Bullet


class Robot(pygame.sprite.Sprite):
    def __init__(self, x, y, color):
        super().__init__()
        self.x = x
        self.y = y
        self.color = color
        self.id = id(self)  # Unique identifier

        # Physical properties
        self.length = 1 # Robot is a square for simplicity

        # Combat properties
        self.shoot_angle = 0  # Direction to shoot
        self.health = 100
        self.alive = True
        self.shoot_cooldown = 0
        self.cooldown_time = 30  # Frames between shots

        # Sensor configuration: circle with range
        self.radar_range = 150

        # Create surface for the robot
        self.image = pygame.Surface((self.length, self.length), pygame.SRCALPHA)
        pygame.draw.rect(self.image, self.color, (0, 0, self.length, self.length))
        self.rect = self.image.get_rect(center=(int(self.x), int(self.y)))

    # TODO: check for collision with other robot
    def update(self, arena_width, arena_height, action=None, other_robot=None):
        """Update robot state based on action.

        action: dict with keys:
            - 'robot_movement': [UP: 0, DOWN: 1, LEFT: 2, RIGHT: 3, NONE: -1]
            - 'turret_rotation': [0, ..., (2*arena_width + 2*arena_height + 1)] -> perimeter values and a NONE option
            - 'shoot': bool for shooting
        other_robot: the other Robot object to check collision against
        """
        if not self.alive:
            return None

        bullet = None

        if action:
            # Movement
            if 'robot_movement' in action:
                new_x, new_y = self.x, self.y
                if action['robot_movement'] == 0:  # UP
                    new_y -= 1
                elif action['robot_movement'] == 1:  # DOWN
                    new_y += 1
                elif action['robot_movement'] == 2:  # LEFT
                    new_x -= 1
                elif action['robot_movement'] == 3:  # RIGHT
                    new_x += 1

                # Only move if not colliding with other robot
                if other_robot is None or (new_x, new_y) != (other_robot.x, other_robot.y):
                    self.x, self.y = new_x, new_y

            # Rotation: turret value maps to discrete angles (0 to N_DIRECTIONS-1)
            # -1 means no rotation (keep current angle)
            N_DIRECTIONS = 16  # Number of discrete aiming directions
            if 'turret' in action and action['turret'] >= 0:
                self.shoot_angle = (action['turret'] / N_DIRECTIONS) * 2 * math.pi

            # Shooting -> NOTE: always shoot when possible for now (shoot will always be True)
            if action.get('shoot', False) and self.shoot_cooldown <= 0:
                bullet = self.shoot()
                self.shoot_cooldown = self.cooldown_time

        # Boundary collision
        self.x = max(0, min(self.x, arena_width))
        self.y = max(0, min(self.y, arena_height))

        # Update rect position
        self.rect.center = (int(self.x), int(self.y))

        # Decrease shoot cooldown
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1

        return bullet

    def shoot(self):
        """Create a bullet traveling in the robot's facing direction."""
        bullet_x = self.x
        bullet_y = self.y
        return Bullet(bullet_x, bullet_y, self.shoot_angle, self.id)

    def take_damage(self, damage):
        """Reduce health when hit."""
        self.health -= damage
        if self.health <= 0:
            self.health = 0
            self.alive = False

    def get_state(self, other_robot, arena_width, arena_height):
        """Get the robot's state as input for AI/neural network.

        Returns a dictionary. The bullets' states are taken from outside.
        """
        # TODO: check for other_robot being alive
        state = {
            'x': self.x / arena_width, # standardize to 0-1
            'y': self.y / arena_height, # standardize to 0-1
            'enemy_x': other_robot.x / arena_width, # standardize to 0-1
            'enemy_y': other_robot.y / arena_height, # standardize to 0-1
            'time_to_shoot': self.shoot_cooldown / self.cooldown_time, 
        }
        return state

    def draw(self, screen, scale=1):
        """Draw the robot on screen."""
        scaled_x = int(self.x * scale)
        scaled_y = int(self.y * scale)
        scaled_size = max(self.length * scale, 4)  # Minimum size for visibility

        if not self.alive:
            # Draw dead robot as gray
            pygame.draw.rect(screen, (100, 100, 100),
                           (scaled_x - scaled_size // 2, scaled_y - scaled_size // 2,
                            scaled_size, scaled_size))
            return

        # Draw robot body -> square
        pygame.draw.rect(screen, self.color,
                        (scaled_x - scaled_size // 2, scaled_y - scaled_size // 2,
                         scaled_size, scaled_size))

        # Draw health bar above robot
        bar_width = scaled_size * 4
        bar_height = max(scaled_size // 2, 4)
        bar_x = scaled_x - bar_width // 2
        bar_y = scaled_y - scaled_size - 10

        # Background (red)
        pygame.draw.rect(screen, (255, 0, 0),
                        (bar_x, bar_y, bar_width, bar_height))
        # Health (green)
        health_width = int(bar_width * (self.health / 100))
        pygame.draw.rect(screen, (0, 255, 0),
                        (bar_x, bar_y, health_width, bar_height))
