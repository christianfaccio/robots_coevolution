import pygame
import math

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
ROBOT_SIZE = 20

class Robot:
    def __init__(self, x, y, angle, energy_points, color=(0, 0, 255), render=True):
        self.render = render
        self.color = color
        self.vel_vector = pygame.math.Vector2(0.8, 0)
        self.vel_vector.rotate_ip(-angle)  # rotate to match initial angle
        self.angle = angle
        self.alive = True
        self.radars = []
        self.energy_points = energy_points
        self.energy = 0
        self.food_collected = 0
        self.distance_traveled = 0
        self.sensor_angles = [0, 72, 144, -144, -72]  # 5 evenly spaced sector centers (72° each)
        self.opponent_range_sensor_length = 100
        self.energy_range_sensor_length = 75

        self.rect = pygame.Rect(0, 0, ROBOT_SIZE, ROBOT_SIZE)
        self.rect.center = (x, y)


    def update(self, left, right, forward, other_robot):
        # Compute movement parameters
        theta_rad = 0.24 * (left - right)  # radians (~0.24 max)
        theta_deg = math.degrees(theta_rad)
        forward_distance = 1.33 * forward  # max 1.33 pixels per step

        # Half turn -> forward -> half turn (paper's combined motion)
        self.rotate(theta_deg / 2)
        self.translate(forward_distance)
        self.rotate(theta_deg / 2)

        # Energy cost in radians (as per paper)
        self.energy -= abs(theta_rad) + forward_distance
        self.collision(other_robot)

    def translate(self, distance):
        if distance > 0:
            direction = self.vel_vector.normalize()
            self.rect.center += direction * distance
            self.distance_traveled += distance

        # if robot is out of screen, keep it inside
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > SCREEN_WIDTH:
            self.rect.right = SCREEN_WIDTH
        if self.rect.top < 0:
            self.rect.top = 0
        if self.rect.bottom > SCREEN_HEIGHT:
            self.rect.bottom = SCREEN_HEIGHT

    def rotate(self, theta):
        '''
        Rotate the robot by theta degrees.
        '''
        self.angle += theta
        self.vel_vector.rotate_ip(-theta)

    def collision(self, other_robot):
        '''
        The robot dies if it collides with another robot.
        '''
        if self.rect.colliderect(other_robot.rect):
            self.alive = False
        # Check collision with energy points
        for energy_point in self.energy_points:
            if self.rect.colliderect(energy_point.rect):
                self.energy += 500 # as paper states
                self.food_collected += 1
                self.energy_points.remove(energy_point)
    
    def _normalize_angle(self, angle):
        '''Normalize angle to [-180, 180].'''
        return (angle + 180) % 360 - 180

    def radar(self, other_robot):
        '''
        Pie-slice sector sensors: 5 sectors of 72° each covering full 360°.
        Each sector reports the normalized distance to the closest target.
        0 = object at robot position, 1 = at max range or not in this sector.

        Returns 11 values:
        - 5 opponent sector distances (normalized)
        - 5 food sector distances (normalized)
        - 1 wall distance (forward, normalized)
        '''
        x, y = self.rect.center
        sector_width = 360.0 / len(self.sensor_angles)  # 72°

        # Angle and distance to opponent
        ox, oy = other_robot.rect.center
        dx_opp = ox - x
        dy_opp = -(oy - y)  # flip y for standard math coords
        dist_to_opp = math.hypot(dx_opp, dy_opp)
        angle_to_opp = math.degrees(math.atan2(dy_opp, dx_opp))
        rel_angle_opp = self._normalize_angle(angle_to_opp - self.angle)

        for center in self.sensor_angles:
            # --- Opponent sensor (unlimited range, normalized by arena diagonal) ---
            angle_diff = self._normalize_angle(rel_angle_opp - center)
            max_dist = math.hypot(SCREEN_WIDTH, SCREEN_HEIGHT)
            if abs(angle_diff) <= sector_width / 2:
                self.radars.append(dist_to_opp / max_dist)
            else:
                self.radars.append(1.0)

            # --- Food sensor (closest energy point in this sector) ---
            closest_food_norm = 1.0
            for ep in self.energy_points:
                ex, ey = ep.rect.center
                dx_food = ex - x
                dy_food = -(ey - y)
                dist_food = math.hypot(dx_food, dy_food)
                if dist_food > self.energy_range_sensor_length:
                    continue
                angle_to_food = math.degrees(math.atan2(dy_food, dx_food))
                rel_angle_food = self._normalize_angle(angle_to_food - self.angle)
                angle_diff_food = self._normalize_angle(rel_angle_food - center)
                if abs(angle_diff_food) <= sector_width / 2:
                    norm_dist = dist_food / self.energy_range_sensor_length
                    closest_food_norm = min(closest_food_norm, norm_dist)
            self.radars.append(closest_food_norm)

        # Single wall sensor (forward-facing)
        forward_rad = math.radians(self.angle)
        dx = math.cos(forward_rad)
        dy = -math.sin(forward_rad)
        length = 0
        while True:
            length += 1
            target_x = int(x + dx * length)
            target_y = int(y + dy * length)
            if target_x < 0 or target_x >= SCREEN_WIDTH or target_y < 0 or target_y >= SCREEN_HEIGHT:
                break
        self.radars.append(length / max(SCREEN_WIDTH, SCREEN_HEIGHT))
    
    def state(self, other_robot):
        '''
        Total of 12 inputs (all normalized):
        - 5 opponent sector distances [0=close, 1=far/not in sector]
        - 5 food sector distances [0=close, 1=far/not in sector]
        - 1 wall distance [0=close, 1=far]
        - energy difference (clamped to [-1, 1])
        '''
        self.radars.clear()
        self.radar(other_robot)

        inputs = list(self.radars)  # 11 normalized inputs
        energy_diff = self.energy - other_robot.energy
        inputs.append(max(-1.0, min(1.0, energy_diff / 1000.0)))

        return inputs
    
    def draw(self, screen):
        pygame.draw.circle(screen, self.color, self.rect.center, ROBOT_SIZE // 2)
        # draw direction white line
        end_pos = (self.rect.center[0] + self.vel_vector.x * ROBOT_SIZE,
                   self.rect.center[1] + self.vel_vector.y * ROBOT_SIZE)
        pygame.draw.line(screen, (255, 255, 255), self.rect.center, end_pos, 2)
        # draw range circles
        pygame.draw.circle(screen, (0, 255, 0), self.rect.center, self.energy_range_sensor_length, 1)
        # draw sector boundaries
        sector_width = 360.0 / len(self.sensor_angles)
        for center in self.sensor_angles:
            boundary_angle = center + sector_width / 2
            rad = math.radians(self.angle + boundary_angle)
            dx = math.cos(rad)
            dy = -math.sin(rad)
            end_pos = (self.rect.center[0] + dx * self.energy_range_sensor_length,
                       self.rect.center[1] + dy * self.energy_range_sensor_length)
            pygame.draw.line(screen, (100, 100, 100), self.rect.center, end_pos, 1)