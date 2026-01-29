import pygame
import random
from .robot import Robot

ARENA_LENGTH = 20
WINDOW_SIZE = 500  # Fixed window size in pixels
SCALE = WINDOW_SIZE / ARENA_LENGTH  # Scale factor for rendering

class Game:
    def __init__(self, render=True):
        self.width = ARENA_LENGTH
        self.height = ARENA_LENGTH
        self.render = render

        if self.render:
            pygame.init()
            self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
            pygame.display.set_caption("Robot War")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)

        self.reset()

    def reset(self):
        """Reset the game to initial state."""

        # Spawn robots ensuring they don't overlap
        x1, y1 = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
        self.robot1 = Robot(x=x1, y=y1, color=(255, 50, 50))

        # Ensure robot2 spawns in a different cell
        while True:
            x2, y2 = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
            if (x2, y2) != (x1, y1):
                break
        self.robot2 = Robot(x=x2, y=y2, color=(50, 50, 255))

        self.bullets = []
        self.game_over = False
        self.winner = None
        self.frame_count = 0
        self.max_frames = 3000  # Max game duration

        return self.get_state()

    def get_state(self):
        """Get full game state for both robots."""
        r1_dict = self.robot1.get_state(self.robot2, self.width, self.height)
        bullets_r2 = [b for b in self.bullets if b.owner_id == self.robot2.id]
        r1_dict['bullets'] = int(bullets_r2 != []) 

        r2_dict = self.robot2.get_state(self.robot1, self.width, self.height)
        bullets_r1 = [b for b in self.bullets if b.owner_id == self.robot1.id]
        r2_dict['bullets'] = int(bullets_r1 != [])

        return {
            'robot1': r1_dict,
            'robot2': r2_dict,
        }

    def step(self, action1=None, action2=None):
        """Advance the game by one frame.

        action1, action2: dicts with 'forward', 'rotate', 'shoot' keys
        Returns: (state, game_over, winner)
        """
        self.frame_count += 1

        # Update robots and collect new bullets (pass other robot to prevent overlap)
        bullet1 = self.robot1.update(self.width, self.height, action1, self.robot2)
        bullet2 = self.robot2.update(self.width, self.height, action2, self.robot1)

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

    def draw(self):
        """Render the current game state."""
        if not self.render:
            return

        # Background
        self.screen.fill((0, 0, 0))

        # Draw grid (offset by half a cell so robots appear centered in cells)
        grid_color = (40, 40, 40)  # Dark gray
        half_cell = SCALE / 2
        for x in range(self.width + 1):
            pygame.draw.line(self.screen, grid_color,
                           (int(x * SCALE + half_cell), 0), (int(x * SCALE + half_cell), WINDOW_SIZE))
        for y in range(self.height + 1):
            pygame.draw.line(self.screen, grid_color,
                           (0, int(y * SCALE + half_cell)), (WINDOW_SIZE, int(y * SCALE + half_cell)))

        # Draw arena border (scaled)
        pygame.draw.rect(self.screen, (0, 255, 0),
                        (0, 0, WINDOW_SIZE, WINDOW_SIZE), 2)

        # Draw robots (pass scale for proper rendering)
        self.robot1.draw(self.screen, SCALE)
        self.robot2.draw(self.screen, SCALE)
        # Draw bullets (pass scale for proper rendering)
        for bullet in self.bullets:
            bullet.draw(self.screen, SCALE)

        # Draw game over message
        if self.game_over:
            if self.winner:
                msg = f"Robot {self.winner} Wins!"
            else:
                msg = "Draw!"
            game_over_text = self.font.render(msg, True, (255, 255, 0))
            text_rect = game_over_text.get_rect(center=(self.width * SCALE // 2, self.height * SCALE // 2))
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