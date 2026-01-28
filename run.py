import random
import math
from game.game import Game

def random_action():
    """Generate a random action for a robot."""
    return {
        'movement': random.uniform(-1, 1) * math.pi,
        'turret': random.uniform(-1, 1) * math.pi,
        'shoot': random.random() < 0.1  
    }


if __name__ == '__main__':

    game = Game()

    print("Press R to restart, ESC to quit")

    running = True
    while running:
        # Handle events
        running = game.handle_events()

        if not game.game_over:
            # Both robots use random actions
            action1 = random_action()
            action2 = random_action()

            # Update game
            game.step(action1, action2)

        # Draw
        game.draw(show_sensors=True)

        # Cap framerate
        game.clock.tick(60)

    game.close()
