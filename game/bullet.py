import pygame
import math

class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y, angle, owner_id, speed=5):
        super().__init__()
        self.x = x
        self.y = y
        self.speed = speed
        self.angle = angle  # Direction the bullet travels (radians)
        self.owner_id = owner_id  # ID of the robot that shot this bullet
        self.alive = True
        self.radius = 1
        self.id = id(self) # Unique identifier

        # Create a surface for the bullet
        self.image = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(self.image, (255, 255, 0), (self.radius, self.radius), self.radius)
        self.rect = self.image.get_rect(center=(int(self.x), int(self.y)))

    def update(self):
        """Move the bullet forward in its direction."""
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed
        self.rect.center = (int(self.x), int(self.y))

    def is_out_of_bounds(self, screen_width, screen_height):
        """Check if bullet has left the arena."""
        return (self.x < 0 or self.x > screen_width or
                self.y < 0 or self.y > screen_height)

    def check_collision(self, robot):
        """Check if bullet hits a robot (not its owner)."""
        if robot.id == self.owner_id:
            return False
        dist = math.hypot(self.x - robot.x, self.y - robot.y)
        return dist < (self.radius + robot.length / 2)

    def draw(self, screen, scale=1):
        """Draw the bullet."""
        scaled_x = int(self.x * scale)
        scaled_y = int(self.y * scale)
        pygame.draw.circle(screen, (255, 255, 0), (scaled_x, scaled_y), int(self.radius))