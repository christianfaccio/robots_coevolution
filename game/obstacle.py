import pygame
import math


class Obstacle:
    def __init__(self, x, y, width, height, color=(128, 128, 128)):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.id = id(self)  # Unique identifier

    def get_rect(self):
        """Return pygame.Rect for convenience."""
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def collides_with_circle(self, cx, cy, radius):
        """Check if a circle collides with this rectangle (AABB-Circle collision)."""
        # Find closest point on rectangle to circle center
        closest_x = max(self.x, min(cx, self.x + self.width))
        closest_y = max(self.y, min(cy, self.y + self.height))

        # Check distance from closest point to circle center
        dist = math.hypot(cx - closest_x, cy - closest_y)
        return dist < radius

    def collides_with_point(self, px, py):
        """Check if a point is inside this rectangle."""
        return (self.x <= px <= self.x + self.width and
                self.y <= py <= self.y + self.height)

    def draw(self, screen):
        """Render the obstacle as a rectangle."""
        pygame.draw.rect(screen, self.color,
                        (int(self.x), int(self.y), int(self.width), int(self.height)))
