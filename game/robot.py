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
        self.sensor_angles = [-150, -60, 0, 60, 150]  # taken from the current angle of the robot
        self.opponent_range_sensor_length = 50
        self.energy_range_sensor_length = 30

        self.rect = pygame.Rect(0, 0, ROBOT_SIZE, ROBOT_SIZE)
        self.rect.center = (x, y)


    def update(self, left, right, forward, other_robot):
        # Compute movement parameters
        theta = math.degrees(0.24 * (left - right))  # ~13.75° max turn
        forward_distance = 1.33 * forward  # max 1.33 pixels per step

        # Half turn -> forward -> half turn (paper's combined motion)
        self.rotate(theta / 2)
        self.translate(forward_distance)
        self.rotate(theta / 2)

        # Energy cost proportional to movement
        self.energy -= abs(theta) + forward_distance
        self.collision(other_robot)

    def translate(self, distance):
        if distance > 0:
            direction = self.vel_vector.normalize()
            self.rect.center += direction * distance

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
                self.energy_points.remove(energy_point)
    
    def radar(self, other_robot):
        '''
        Measure distances (paper: 5 opponent + 5 food + 1 wall = 11 sensors).
        Returns:
        - 5 distances to opponent (0 if no opponent in sensor range)
        - 5 distances to energy points (0 if no energy point in sensor range)
        - 1 distance to wall (forward-facing only)
        '''
        x = int(self.rect.center[0])
        y = int(self.rect.center[1])

        for sensor_angle in self.sensor_angles:
            # Opponent radar
            length = 0
            sensor_rad = math.radians(self.angle + sensor_angle)
            dx = math.cos(sensor_rad)
            dy = -math.sin(sensor_rad)

            while length < self.opponent_range_sensor_length:
                length += 1
                target_x = int(x + dx * length)
                target_y = int(y + dy * length)

                if target_x < 0 or target_x >= SCREEN_WIDTH or target_y < 0 or target_y >= SCREEN_HEIGHT:
                    break

                if other_robot.rect.collidepoint(target_x, target_y):
                    break

            self.radars.append(length if length < self.opponent_range_sensor_length else 0)

            # Energy radar
            length = 0
            while length < self.energy_range_sensor_length:
                length += 1
                target_x = int(x + dx * length)
                target_y = int(y + dy * length)

                if target_x < 0 or target_x >= SCREEN_WIDTH or target_y < 0 or target_y >= SCREEN_HEIGHT:
                    break

                hit_energy = False
                for energy_point in self.energy_points:
                    if energy_point.rect.collidepoint(target_x, target_y):
                        hit_energy = True
                        break
                if hit_energy:
                    break

            self.radars.append(length if length < self.energy_range_sensor_length else 0)

        # Single wall sensor (forward-facing, as per paper)
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
        self.radars.append(length)
    
    def state(self, other_robot):
        '''  
        Total of 12 inputs:
        - 5 sensors distances to opponent
        - 5 sensors distances to energy points
        - 1 sensor distance to wall
        - difference in energy levels
        '''
        self.radars.clear()
        self.radar(other_robot)
        
        inputs = []
        inputs.extend([radar for radar in self.radars])  # 11 inputs from radars
        energy_diff = self.energy - other_robot.energy
        inputs.append(energy_diff)  # 1 input for energy difference

        return inputs
    
    def draw(self, screen):
        pygame.draw.circle(screen, self.color, self.rect.center, ROBOT_SIZE // 2)
        # draw direction white line
        end_pos = (self.rect.center[0] + self.vel_vector.x * ROBOT_SIZE,
                   self.rect.center[1] + self.vel_vector.y * ROBOT_SIZE)
        pygame.draw.line(screen, (255, 255, 255), self.rect.center, end_pos, 2)
        # draw circle for radar range
        pygame.draw.circle(screen, (255, 255, 255), self.rect.center, self.opponent_range_sensor_length, 1)
        pygame.draw.circle(screen, (0, 255, 0), self.rect.center, self.energy_range_sensor_length, 1)
        # draw sensors
        for sensor_angle in self.sensor_angles:
            sensor_rad = math.radians(self.angle + sensor_angle)
            dx = math.cos(sensor_rad)
            dy = -math.sin(sensor_rad)
            end_pos = (self.rect.center[0] + dx * self.opponent_range_sensor_length,
                       self.rect.center[1] + dy * self.opponent_range_sensor_length)
            pygame.draw.circle(screen, (255, 255, 255), end_pos, 3)
            end_pos_energy = (self.rect.center[0] + dx * self.energy_range_sensor_length,
                       self.rect.center[1] + dy * self.energy_range_sensor_length)
            pygame.draw.circle(screen, (0, 255, 0), end_pos_energy, 3)