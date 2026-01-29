import random
from game.game import Game, ARENA_LENGTH

N_DIRECTIONS = 16  # Must match robot.py

def random_action():
    """Generate a random action for a robot."""
    return {
        'robot_movement': random.randint(-1, 3),  # -1=none, 0=up, 1=down, 2=left, 3=right
        'turret': random.randint(-1, N_DIRECTIONS - 1),  # -1=no change, 0 to 15 = direction
        'shoot': True
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
        game.draw()

        # Cap framerate
        game.clock.tick(10)

    game.close()
