import pygame
import random
import sys
import numpy as np
import os

# Set the Pygame window position
os.environ['SDL_VIDEO_WINDOW_POS'] = '100,100'  # Adjust these values as needed

class GraphicalMazeGenerator:
    def __init__(self, width, height):
        if not (6 <= width <= 15 and 6 <= height <= 15):
            raise ValueError("Maze dimensions must be between 6x6 and 15x15")
        
        # Initialize Pygame
        pygame.init()
        
        # Maze properties
        self.width = width
        self.height = height
        self.cell_size = 30  # Further reduced cell size (from 40 to 30)
        self.wall_thickness = 2  # Reduced wall thickness
        
        # Calculate window size
        self.window_width = (2 * width + 1) * self.cell_size
        self.window_height = (2 * height + 1) * self.cell_size
        
        # Set up display
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Random Maze Generator with Q-learning")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.TRAIL_COLOR = (200, 200, 200)  # Light gray for the trail
        
        # Initialize maze data
        self.maze = [['#' for _ in range(2 * width + 1)] 
                    for _ in range(2 * height + 1)]
        self.visited = [[False for _ in range(width)] 
                       for _ in range(height)]

        # Q-learning parameters
        self.q_table = {}  # Q-table: key = (state), value = [Q-values for actions]
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01

        # Actions: up, down, left, right
        self.actions = [0, 1, 2, 3]
        self.action_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def generate(self):
        # Reset maze and visited data
        self.maze = [['#' for _ in range(2 * self.width + 1)] 
                    for _ in range(2 * self.height + 1)]
        self.visited = [[False for _ in range(self.width)] 
                       for _ in range(self.height)]
        
        # Generate new maze with more complexity
        self._generate_complex_maze(0, 0)
        self.maze[1][0] = ' '  # Entrance
        self.maze[2 * self.height - 1][2 * self.width] = ' '  # Exit
        self.draw_maze()
        return self.maze

    def _generate_complex_maze(self, x, y):
        self.visited[y][x] = True
        maze_x = 2 * x + 1
        maze_y = 2 * y + 1
        self.maze[maze_y][maze_x] = ' '
        
        # Draw current state
        self.draw_maze()
        pygame.time.delay(50)  # Reduced delay for faster generation
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            
            if (0 <= new_x < self.width and 
                0 <= new_y < self.height and 
                not self.visited[new_y][new_x]):
                
                wall_x = maze_x + dx
                wall_y = maze_y + dy
                self.maze[wall_y][wall_x] = ' '
                self._generate_complex_maze(new_x, new_y)
                
                # Introduce loops by occasionally revisiting cells
                if random.random() < 0.2:  # 20% chance to create a loop
                    self._generate_complex_maze(new_x, new_y)

    def draw_maze(self):
        self.screen.fill(self.WHITE)
        
        # Draw maze cells
        for y in range(len(self.maze)):
            for x in range(len(self.maze[0])):
                cell_x = x * self.cell_size
                cell_y = y * self.cell_size
                
                if self.maze[y][x] == '#':
                    pygame.draw.rect(self.screen, self.BLACK,
                                  (cell_x, cell_y, self.cell_size, self.cell_size))
                
        # Mark entrance and exit
        pygame.draw.rect(self.screen, self.GREEN,
                       (0, self.cell_size, self.cell_size, self.cell_size))
        pygame.draw.rect(self.screen, self.RED,
                       (self.window_width - self.cell_size,
                        self.window_height - 2 * self.cell_size,
                        self.cell_size, self.cell_size))
        
        pygame.display.flip()

    def get_state(self, x, y):
        # Convert (x, y) position to a state identifier
        return (x, y)

    def get_reward(self, x, y):
        # Check if the agent has reached the exit
        if x == 2 * self.width and y == 2 * self.height - 1:
            return 100  # Large reward for reaching the exit
        elif self.maze[y][x] == '#':
            return -10  # Penalty for hitting a wall
        else:
            return -1  # Small penalty for each step

    def is_valid_move(self, x, y):
        # Check if the new position is within the maze bounds and is a valid cell (not a wall)
        return (0 <= x < len(self.maze[0]) and 
                0 <= y < len(self.maze) and 
                self.maze[y][x] == ' ')

    def q_learning(self, episodes=1000):
        # Reset Q-table for new maze
        self.q_table = {}
        self.epsilon = 1.0  # Reset exploration rate

        for episode in range(episodes):
            # Reset agent to the starting position
            state = self.get_state(1, 1)  # Start at the entrance
            done = False

            while not done:
                # Choose action (epsilon-greedy)
                if random.uniform(0, 1) < self.epsilon:
                    action = random.choice(self.actions)  # Explore
                else:
                    if state in self.q_table:
                        action = np.argmax(self.q_table[state])  # Exploit
                    else:
                        action = random.choice(self.actions)

                # Perform action
                dx, dy = self.action_directions[action]
                new_x = state[0] + dx
                new_y = state[1] + dy

                # Check if the new position is valid
                if self.is_valid_move(new_x, new_y):
                    new_state = self.get_state(new_x, new_y)
                    reward = self.get_reward(new_x, new_y)

                    # Update Q-value
                    if state not in self.q_table:
                        self.q_table[state] = [0] * len(self.actions)
                    if new_state not in self.q_table:
                        self.q_table[new_state] = [0] * len(self.actions)

                    old_value = self.q_table[state][action]
                    next_max = np.max(self.q_table[new_state])
                    new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
                    self.q_table[state][action] = new_value

                    # Move to the new state
                    state = new_state

                    # Check if the episode is done
                    if reward == 100:
                        done = True
                else:
                    # Invalid move, stay in the same state
                    reward = -10
                    if state not in self.q_table:
                        self.q_table[state] = [0] * len(self.actions)
                    self.q_table[state][action] += self.alpha * (reward - self.q_table[state][action])

            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def visualize_path(self):
        # Start at the entrance
        state = self.get_state(1, 1)
        path = [state]

        # Draw the initial maze
        self.draw_maze()

        # Draw the character (a small circle)
        character_radius = self.cell_size // 4
        trail_positions = set()

        while state != self.get_state(2 * self.width, 2 * self.height - 1):
            if state in self.q_table:
                action = np.argmax(self.q_table[state])
                dx, dy = self.action_directions[action]
                new_x = state[0] + dx
                new_y = state[1] + dy

                # Ensure the new position is valid
                if self.is_valid_move(new_x, new_y):
                    # Draw the trail
                    trail_positions.add((state[0] * self.cell_size + self.cell_size // 2,
                                        state[1] * self.cell_size + self.cell_size // 2))
                    for (tx, ty) in trail_positions:
                        pygame.draw.circle(self.screen, self.TRAIL_COLOR, (tx, ty), character_radius // 2)

                    # Move to the new state
                    state = self.get_state(new_x, new_y)
                    path.append(state)

                    # Draw the character
                    char_x = state[0] * self.cell_size + self.cell_size // 2
                    char_y = state[1] * self.cell_size + self.cell_size // 2
                    pygame.draw.circle(self.screen, self.BLUE, (char_x, char_y), character_radius)

                    pygame.display.flip()
                    pygame.time.delay(100)  # Reduced delay for faster visualization
                else:
                    break
            else:
                break

def main():
    # Initialize with random dimensions
    width = random.randint(6, 15)
    height = random.randint(6, 15)
    
    print(f"Generating a {width}x{height} maze...")
    generator = GraphicalMazeGenerator(width, height)
    generator.generate()
    
    # Train the agent using Q-learning
    generator.q_learning(episodes=1000)
    
    # Visualize the learned path
    generator.visualize_path()
    
    # Keep window open until closed
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Generate new dimensions and maze
                    width = random.randint(6, 15)
                    height = random.randint(6, 15)
                    print(f"Generating a {width}x{height} maze...")
                    generator = GraphicalMazeGenerator(width, height)
                    generator.generate()
                    generator.q_learning(episodes=1000)
                    generator.visualize_path()  
                elif event.key == pygame.K_ESCAPE:
                    running = False
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()