import pygame
import random
from robot import Robot
from obstacle import Obstacle

class Game:
    def __init__(self, width=1000, height=1000, render=True):
        self.width = width
        self.height = height
        self.render = render

        if self.render:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Robot War")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)

        self.reset()

    def reset(self):
        """Reset the game to initial state."""
        # Spawn obstacles first
        self.obstacles = []
        self._spawn_obstacles()

        # Spawn robots ensuring they don't overlap with obstacles
        self.robot1 = self._spawn_robot(color=(255, 50, 50))
        self.robot2 = self._spawn_robot(color=(50, 50, 255))

        self.bullets = []
        self.game_over = False
        self.winner = None
        self.frame_count = 0
        self.max_frames = 3000  # Max game duration

        return self.get_state()

    def _spawn_robot(self, color):
        """Spawn a robot ensuring it doesn't overlap with obstacles."""
        margin = 20
        for _ in range(100):  # Max attempts
            x = random.randint(margin, self.width - margin)
            y = random.randint(margin, self.height - margin)

            # Check no overlap with obstacles
            overlaps = False
            for obstacle in self.obstacles:
                if obstacle.collides_with_circle(x, y, 15):  # Robot radius + buffer
                    overlaps = True
                    break

            if not overlaps:
                return Robot(x=x, y=y, color=color)

        # Fallback: spawn anyway
        return Robot(x=random.randint(margin, self.width - margin),
                    y=random.randint(margin, self.height - margin),
                    color=color)

    def _spawn_obstacles(self):
        """Generate random obstacles in the arena."""
        num_obstacles = random.randint(3, 8)
        margin = 50  # Keep away from edges

        for _ in range(num_obstacles):
            for attempt in range(100):  # Max attempts to find valid position
                width = random.randint(50, 150)
                height = random.randint(50, 150)
                x = random.randint(margin, self.width - margin - width)
                y = random.randint(margin, self.height - margin - height)

                obstacle = Obstacle(x, y, width, height)

                # Check no overlap with other obstacles
                if not self._overlaps_obstacles(obstacle):
                    self.obstacles.append(obstacle)
                    break

    def _overlaps_obstacles(self, new_obstacle):
        """Check if new obstacle overlaps with existing obstacles."""
        new_rect = new_obstacle.get_rect()
        buffer = 20  # Minimum gap between obstacles

        for existing in self.obstacles:
            existing_rect = existing.get_rect()
            # Inflate rect to add buffer
            inflated = existing_rect.inflate(buffer, buffer)
            if new_rect.colliderect(inflated):
                return True
        return False

    def get_state(self):
        """Get full game state for both robots."""
        # Update sensors with obstacle awareness
        self.robot1.update_sensors(self.robot2, self.width, self.height, obstacles=self.obstacles)
        self.robot2.update_sensors(self.robot1, self.width, self.height, obstacles=self.obstacles)

        return {
            'robot1': self.robot1.get_state(self.robot2, self.width, self.height),
            'robot2': self.robot2.get_state(self.robot1, self.width, self.height),
        }

    def step(self, action1=None, action2=None):
        """Advance the game by one frame.

        action1, action2: dicts with 'forward', 'rotate', 'shoot' keys
        Returns: (state, game_over, winner)
        """
        self.frame_count += 1

        # Update robots and collect new bullets (with obstacle awareness)
        bullet1 = self.robot1.update(self.width, self.height, action1, self.obstacles)
        bullet2 = self.robot2.update(self.width, self.height, action2, self.obstacles)

        if bullet1:
            self.bullets.append(bullet1)
        if bullet2:
            self.bullets.append(bullet2)

        # Update bullets
        bullets_to_remove = []
        for bullet in self.bullets:
            bullet.update()

            # Check if bullet hit robot1
            if bullet.check_collision(self.robot1):
                self.robot1.take_damage(20)
                bullets_to_remove.append(bullet)
            # Check if bullet hit robot2
            elif bullet.check_collision(self.robot2):
                self.robot2.take_damage(20)
                bullets_to_remove.append(bullet)
            # Check if bullet hit obstacle
            else:
                hit_obstacle = False
                for obstacle in self.obstacles:
                    if bullet.check_obstacle_collision(obstacle):
                        hit_obstacle = True
                        break
                if hit_obstacle:
                    bullets_to_remove.append(bullet)
                # Check if bullet left arena
                elif bullet.is_out_of_bounds(self.width, self.height):
                    bullets_to_remove.append(bullet)

        for bullet in bullets_to_remove:
            self.bullets.remove(bullet)

        # Check win conditions
        if not self.robot1.alive and not self.robot2.alive:
            self.game_over = True
            self.winner = None  # Draw
        elif not self.robot1.alive:
            self.game_over = True
            self.winner = 2
        elif not self.robot2.alive:
            self.game_over = True
            self.winner = 1
        elif self.frame_count >= self.max_frames:
            self.game_over = True
            # Winner by health
            if self.robot1.health > self.robot2.health:
                self.winner = 1
            elif self.robot2.health > self.robot1.health:
                self.winner = 2
            else:
                self.winner = None  # Draw

        state = self.get_state()
        return state, self.game_over, self.winner

    def draw(self, show_sensors=True):
        """Render the current game state."""
        if not self.render:
            return

        # Background
        self.screen.fill((0, 0, 0))

        # Draw arena border
        pygame.draw.rect(self.screen, (0, 255, 0),
                        (0, 0, self.width, self.height), 1)

        # Draw obstacles (before robots/bullets so they appear behind)
        for obstacle in self.obstacles:
            obstacle.draw(self.screen)

        # Draw sensors (before robots so they appear behind)
        if show_sensors:
            self.robot1.update_sensors(self.robot2, self.width, self.height, self.screen, self.obstacles)
            self.robot2.update_sensors(self.robot1, self.width, self.height, self.screen, self.obstacles)

        # Draw robots
        self.robot1.draw(self.screen)
        self.robot2.draw(self.screen)

        # Draw bullets
        for bullet in self.bullets:
            bullet.draw(self.screen)

        # Draw game over message
        if self.game_over:
            if self.winner:
                msg = f"Robot {self.winner} Wins!"
            else:
                msg = "Draw!"
            game_over_text = self.font.render(msg, True, (255, 255, 0))
            text_rect = game_over_text.get_rect(center=(self.width // 2, self.height // 2))
            self.screen.blit(game_over_text, text_rect)

        pygame.display.flip()

    def handle_events(self):
        """Handle pygame events. Returns False if should quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_r:
                    self.reset()
        return True

    def close(self):
        """Clean up pygame."""
        if self.render:
            pygame.quit()