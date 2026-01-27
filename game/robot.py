import pygame
import math
import random
from bullet import Bullet


class Robot(pygame.sprite.Sprite):
    def __init__(self, x, y, color, robot_id):
        super().__init__()
        self.x = x
        self.y = y
        self.angle = random.uniform(-1,1) * math.pi  
        self.shoot_angle = random.uniform(-1,1) * math.pi  
        self.color = color
        self.robot_id = robot_id

        # Physical properties
        self.radius = 10
        self.speed = 0
        self.max_speed = 1

        # Combat properties
        self.health = 100
        self.alive = True
        self.shoot_cooldown = 0
        self.cooldown_time = 30  # Frames between shots

        # Sensor configuration: angles relative to robot facing direction
        self.sensor_angles = [math.radians(a) for a in range(0, 361, 45)]  # 0 to 360 degrees in 45 deg steps
        self.sensor_range = 150
        self.sensor_readings = [0] * len(self.sensor_angles)

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
        return Bullet(bullet_x, bullet_y, self.shoot_angle, self.robot_id)

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

        Returns normalized readings [0, 1] where 1 = max range (nothing detected)
        """
        self.sensor_readings = []

        for sensor_angle_offset in self.sensor_angles:
            sensor_angle = self.angle + sensor_angle_offset

            # Cast ray and find closest intersection
            min_dist = self.sensor_range

            # Check distance to walls
            wall_dist = self._ray_to_walls(sensor_angle, arena_width, arena_height)
            if wall_dist < min_dist:
                min_dist = wall_dist

            # Check distance to enemy robot
            if other_robot and other_robot.alive:
                robot_dist = self._ray_to_circle(
                    sensor_angle, other_robot.x, other_robot.y, other_robot.radius
                )
                if robot_dist and robot_dist < min_dist:
                    min_dist = robot_dist

            # Check distance to obstacles
            if obstacles:
                for obstacle in obstacles:
                    obstacle_dist = self._ray_to_obstacle(sensor_angle, obstacle)
                    if obstacle_dist and obstacle_dist < min_dist:
                        min_dist = obstacle_dist

            # Normalize to [0, 1]
            normalized = min_dist / self.sensor_range
            self.sensor_readings.append(normalized)

            # Draw sensor rays if screen provided
            if screen:
                end_x = self.x + math.cos(sensor_angle) * min_dist
                end_y = self.y + math.sin(sensor_angle) * min_dist
                pygame.draw.line(screen, (100, 100, 100),
                               (int(self.x), int(self.y)),
                               (int(end_x), int(end_y)), 1)
                pygame.draw.circle(screen, (0, 255, 0), (int(end_x), int(end_y)), 3)

        return self.sensor_readings

    def _ray_to_walls(self, angle, arena_width, arena_height):
        """Calculate distance from robot to arena walls along a ray."""
        dx = math.cos(angle)
        dy = math.sin(angle)

        distances = []

        # Distance to right wall (x = arena_width)
        if dx > 0:
            t = (arena_width - self.x) / dx
            distances.append(t)

        # Distance to left wall (x = 0)
        if dx < 0:
            t = -self.x / dx
            distances.append(t)

        # Distance to bottom wall (y = arena_height)
        if dy > 0:
            t = (arena_height - self.y) / dy
            distances.append(t)

        # Distance to top wall (y = 0)
        if dy < 0:
            t = -self.y / dy
            distances.append(t)

        return min(distances) if distances else self.sensor_range

    def _ray_to_circle(self, angle, cx, cy, radius):
        """Calculate distance from robot to a circle (other robot) along a ray.

        Uses ray-circle intersection formula.
        """
        dx = math.cos(angle)
        dy = math.sin(angle)

        # Vector from ray origin to circle center
        fx = self.x - cx
        fy = self.y - cy

        a = dx * dx + dy * dy  # Always 1 for unit direction
        b = 2 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - radius * radius

        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return None  # No intersection

        discriminant = math.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)

        # Return closest positive intersection
        if t1 >= 0:
            return t1
        if t2 >= 0:
            return t2
        return None

    def _ray_to_obstacle(self, angle, obstacle):
        """Calculate distance from robot to obstacle along a ray.

        Uses ray-AABB intersection (parametric line equation).
        Returns distance to obstacle or None if no intersection.
        """
        dx = math.cos(angle)
        dy = math.sin(angle)

        # Avoid division by zero
        if dx == 0:
            dx = 1e-10
        if dy == 0:
            dy = 1e-10

        # Calculate t values for intersection with obstacle edges
        t1 = (obstacle.x - self.x) / dx
        t2 = (obstacle.x + obstacle.width - self.x) / dx
        t3 = (obstacle.y - self.y) / dy
        t4 = (obstacle.y + obstacle.height - self.y) / dy

        # Find the intersection interval
        tmin = max(min(t1, t2), min(t3, t4))
        tmax = min(max(t1, t2), max(t3, t4))

        # No intersection if tmax < 0 or tmin > tmax
        if tmax < 0 or tmin > tmax:
            return None

        # Return closest positive intersection
        if tmin >= 0:
            return tmin
        return None

    def get_state(self, other_robot, arena_width, arena_height):
        """Get the robot's state as input for AI/neural network.

        Returns a list of normalized values:
        - 5 sensor readings [0, 1]
        - Distance to enemy (normalized)
        - Angle to enemy (normalized to [-1, 1])
        - Own health (normalized)
        - Enemy health (normalized)
        """
        state = list(self.sensor_readings)

        if other_robot and other_robot.alive:
            # Distance to enemy
            dist = math.hypot(other_robot.x - self.x, other_robot.y - self.y)
            max_dist = math.hypot(arena_width, arena_height)
            state.append(dist / max_dist)

            # Angle to enemy relative to robot's facing direction
            angle_to_enemy = math.atan2(other_robot.y - self.y, other_robot.x - self.x)
            relative_angle = angle_to_enemy - self.angle
            # Normalize to [-pi, pi]
            while relative_angle > math.pi:
                relative_angle -= 2 * math.pi
            while relative_angle < -math.pi:
                relative_angle += 2 * math.pi
            state.append(relative_angle / math.pi)

            state.append(other_robot.health / 100)
        else:
            state.extend([1.0, 0.0, 0.0])

        state.append(self.health / 100)

        return state

    def draw(self, screen):
        """Draw the robot on screen."""
        if not self.alive:
            # Draw dead robot as gray
            pygame.draw.circle(screen, (100, 100, 100), (int(self.x), int(self.y)), self.radius)
            return

        # Draw robot body
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

        # Draw direction indicator (gun barrel)
        # barrel_length = self.radius * 1.5
        # end_x = self.x + math.cos(self.shoot_angle) * barrel_length
        # end_y = self.y + math.sin(self.angle) * barrel_length
        # pygame.draw.line(screen, (255, 255, 255),
        #                 (int(self.x), int(self.y)),
        #                 (int(end_x), int(end_y)), 3)

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
