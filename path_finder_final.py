import pygame
import sys
from queue import Queue, PriorityQueue
import random
import time
from collections import deque

# Initialize Pygame
pygame.init()
pygame.font.init()

# Constants
INIT_WIDTH, INIT_HEIGHT = 800, 850
MIN_WIDTH, MIN_HEIGHT = 400, 450
PANEL_HEIGHT = 100
CONTROLS_HEIGHT = 50
rows, cols = 20, 20
obstacles = int(0.3 * rows * cols)

BUTTON_COLOR = (70, 130, 180)  # Steel blue
BUTTON_HOVER = (100, 150, 200)  # Lighter blue
BUTTON_TEXT = (255,255,255) #white

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (200, 200, 200)
LIGHT_BLUE = (173, 216, 230)
PURPLE = (128, 0, 128)

# Fonts
font = pygame.font.SysFont('Arial', 16)
large_font = pygame.font.SysFont('Arial', 24)

class PathfinderApp:
    def __init__(self):
        self.width, self.height = INIT_WIDTH, INIT_HEIGHT
        self.cell_len = self.width // cols
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption("Pathfinder's Quest!")
        
        self.grid = [[0 for _ in range(cols)] for _ in range(rows)]  # 0=empty, 1=obstacle, 2=start, 3=goal
        self._setup_grid()
        self.reset_state()

    def _setup_grid(self):
        # Place obstacles
        for _ in range(obstacles):
            while True:
                row, col = random.randint(0, rows-1), random.randint(0, cols-1)
                if self.grid[row][col] == 0:
                    self.grid[row][col] = 1
                    break
        
        # Place start and goal
        while True:
            self.start = (random.randint(0, rows-1), random.randint(0, cols-1))
            self.goal = (random.randint(0, rows-1), random.randint(0, cols-1))
            if (self.grid[self.start[0]][self.start[1]] != 1 and 
                self.grid[self.goal[0]][self.goal[1]] != 1 and 
                self.start != self.goal):
                self.grid[self.start[0]][self.start[1]] = 2
                self.grid[self.goal[0]][self.goal[1]] = 3
                break

    def reset_state(self):
        self.path = None
        self.visited_order = []
        self.algorithm = None
        self.time_taken = 0
        self.visited_count = 0

    def handle_resize(self, new_width, new_height):
        self.width = max(new_width, MIN_WIDTH)
        self.height = max(new_height, MIN_HEIGHT)
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        self.cell_len = min(self.width // cols, (self.height - PANEL_HEIGHT - CONTROLS_HEIGHT) // rows)

    def draw_grid(self):
        for row in range(rows):
            for col in range(cols):
                color = WHITE
                if self.grid[row][col] == 1:
                    color = BLACK
                elif self.grid[row][col] == 2:
                    color = GREEN
                elif self.grid[row][col] == 3:
                    color = RED
                
                pygame.draw.rect(self.screen, color, 
                               (col * self.cell_len, row * self.cell_len, 
                                self.cell_len, self.cell_len))
                pygame.draw.rect(self.screen, BLACK, 
                               (col * self.cell_len, row * self.cell_len, 
                                self.cell_len, self.cell_len), 1)

    def draw_visited(self):
        for (row, col) in self.visited_order:
            if self.grid[row][col] not in [2, 3]:  # Don't overwrite start/goal
                pygame.draw.rect(self.screen, LIGHT_BLUE, 
                               (col * self.cell_len, row * self.cell_len, 
                                self.cell_len, self.cell_len))
                pygame.draw.rect(self.screen, BLACK, 
                               (col * self.cell_len, row * self.cell_len, 
                                self.cell_len, self.cell_len), 1)

    def draw_path(self):
        if self.path:
            for (row, col) in self.path:
                if self.grid[row][col] not in [2, 3]:  # Don't overwrite start/goal
                    pygame.draw.rect(self.screen, BLUE, 
                                   (col * self.cell_len, row * self.cell_len, 
                                    self.cell_len, self.cell_len))
                    pygame.draw.rect(self.screen, BLACK, 
                                   (col * self.cell_len, row * self.cell_len, 
                                    self.cell_len, self.cell_len), 1)

    def draw_info_panel(self):
        panel_y = rows * self.cell_len
    
        # Ensure panel stays visible
        if panel_y + PANEL_HEIGHT + CONTROLS_HEIGHT > self.height:
            panel_y = self.height - PANEL_HEIGHT - CONTROLS_HEIGHT
    
        # Draw panel background (metrics)
        pygame.draw.rect(self.screen, GRAY, (0, panel_y, self.width, PANEL_HEIGHT))
    
        # Prepare and display metrics
        algorithm_text = f"Algorithm: {self.algorithm}" if self.algorithm else "Select Algorithm"
        time_text = f"Time: {self.time_taken:.4f}s" if self.time_taken else "Time: -"
        path_text = f"Path: {len(self.path)} steps" if self.path else "Path: -"
        visited_text = f"Visited: {self.visited_count} nodes"
    
        metrics = [
            font.render(algorithm_text, True, BLACK),
            font.render(time_text, True, BLACK),
            font.render(path_text, True, BLACK),
            font.render(visited_text, True, BLACK)
        ]
    
        # Display metrics in a row
        num_items = len(metrics)
        for i, metric in enumerate(metrics):
            x_pos = 20 + i * (self.width - 40) // num_items
            self.screen.blit(metric, (x_pos, panel_y + PANEL_HEIGHT//2 - metric.get_height()//2))
        
        # Draw controls panel below metrics
        controls_y = panel_y + PANEL_HEIGHT
        pygame.draw.rect(self.screen, LIGHT_BLUE, (0, controls_y, self.width, CONTROLS_HEIGHT))
        
        # Draw control buttons
        button_width = 80
        button_height = 30
        spacing = 10
        start_x = (self.width - (5*button_width + 4*spacing)) // 2  # Center buttons
        
        control_buttons = [
            ("BFS", pygame.K_b),
            ("DFS", pygame.K_d),
            ("A*", pygame.K_a),
            ("Compare", pygame.K_c),
            ("Reset", pygame.K_r)
        ]
        
        for i, (text, _) in enumerate(control_buttons):
            rect = pygame.Rect(
                start_x + i*(button_width + spacing),
                controls_y + (CONTROLS_HEIGHT - button_height)//2,
                button_width,
                button_height
            )
            
            # Draw button
            pygame.draw.rect(self.screen, BUTTON_COLOR, rect, border_radius=5)
            pygame.draw.rect(self.screen, BLACK, rect, 2, border_radius=5)
            
            # Draw text
            text_surf = font.render(text, True, WHITE)
            text_rect = text_surf.get_rect(center=rect.center)
            self.screen.blit(text_surf, text_rect)
        
        # Draw separator
        pygame.draw.line(self.screen, BLACK, (0, panel_y), (self.width, panel_y), 2)
        pygame.draw.line(self.screen, BLACK, (0, controls_y), (self.width, controls_y), 1)

    def bfs(self, start, goal, visualize=False):
        start_time = time.time()
        queue = Queue()
        queue.put((start, [start]))
        visited = set()
        visited_order = []
        
        while not queue.empty():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            (current, path) = queue.get()
            visited.add(current)
            visited_order.append(current)
            
            if visualize:
                self.screen.fill(WHITE)
                self.draw_grid()
                self.visited_order = visited_order
                self.path = path 
                self.draw_visited()
                self.draw_path()
                self.draw_info_panel()
                pygame.display.flip()
                pygame.time.delay(50)
            
            if current == goal:
                return path, time.time() - start_time, len(visited_order)
            
            for (dr, dc) in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                neighbor = (current[0] + dr, current[1] + dc)
                if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                    if self.grid[neighbor[0]][neighbor[1]] != 1 and neighbor not in visited:
                        visited.add(neighbor)
                        queue.put((neighbor, path + [neighbor]))
        
        return None, time.time() - start_time, len(visited_order)

    def dfs(self, start, goal, visualize=False):
        start_time = time.time()
        stack = [(start, [start])]
        visited = set()
        visited_order = []
        
        while stack:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            (current, path) = stack.pop()
            visited.add(current)
            visited_order.append(current)
            
            if visualize:
                self.screen.fill(WHITE)
                self.draw_grid()
                self.visited_order = visited_order
                self.path = path 
                self.draw_visited()
                self.draw_path()
                self.draw_info_panel()
                pygame.display.flip()
                pygame.time.delay(50)
            
            if current == goal:
                return path, time.time() - start_time, len(visited_order)
            
            # Reverse the order to explore in consistent direction
            for (dr, dc) in reversed([(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]):
                neighbor = (current[0] + dr, current[1] + dc)
                if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                    if self.grid[neighbor[0]][neighbor[1]] != 1 and neighbor not in visited:
                        visited.add(neighbor)
                        stack.append((neighbor, path + [neighbor]))
        
        return None, time.time() - start_time, len(visited_order)

    def a_star(self, start, goal, visualize=False):
        start_time = time.time()
        
        def heuristic(a, b):
            # Euclidean distance
            return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5
        
        pq = PriorityQueue()
        pq.put((0, start, [start]))
        visited = set()
        visited_order = []
        cost_so_far = {start: 0}
        
        while not pq.empty():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            (priority, current, path) = pq.get()
            visited.add(current)
            visited_order.append(current)
            
            if visualize:
                self.screen.fill(WHITE)
                self.draw_grid()
                self.visited_order = visited_order
                self.path = path 
                self.draw_visited()
                self.draw_path()
                self.draw_info_panel()
                pygame.display.flip()
                pygame.time.delay(50)
            
            if current == goal:
                return path, time.time() - start_time, len(visited_order)
            
            for (dr, dc) in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                neighbor = (current[0] + dr, current[1] + dc)
                if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                    if self.grid[neighbor[0]][neighbor[1]] != 1:
                        new_cost = cost_so_far[current] + 1
                        if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                            cost_so_far[neighbor] = new_cost
                            priority = new_cost + heuristic(neighbor, goal)
                            pq.put((priority, neighbor, path + [neighbor]))
        
        return None, time.time() - start_time, len(visited_order)

    def compare_algorithms(self):
        results = []
        
        # Run each algorithm without visualization for accurate timing
        path, time_taken, visited = self.bfs(self.start, self.goal, False)
        results.append(("BFS", time_taken, len(path) if path else 0, visited))
        
        path, time_taken, visited = self.dfs(self.start, self.goal, False)
        results.append(("DFS", time_taken, len(path) if path else 0, visited))
        
        path, time_taken, visited = self.a_star(self.start, self.goal, False)
        results.append(("A*", time_taken, len(path) if path else 0, visited))
        
        # Find the best in each category
        best_time = min(results, key=lambda x: x[1])[0]
        best_path = min(results, key=lambda x: x[2])[0] if any(x[2] > 0 for x in results) else "N/A"
        best_visited = min(results, key=lambda x: x[3])[0]
        
        # Display comparison
        self.screen.fill(WHITE)
        title = large_font.render("Algorithm Comparison Results", True, BLACK)
        self.screen.blit(title, (self.width//2 - title.get_width()//2, 20))
        
        y_pos = 70
        for name, t, pl, vis in results:
            text = font.render(f"{name}: Time={t:.4f}s, Path={pl}, Visited={vis}", True, BLACK)
            self.screen.blit(text, (self.width//2 - text.get_width()//2, y_pos))
            y_pos += 30
        
        conclusion = large_font.render(f"Best: Time={best_time}, Path={best_path}, Visited={best_visited}", True, BLUE)
        self.screen.blit(conclusion, (self.width//2 - conclusion.get_width()//2, y_pos + 40))
        
        pygame.display.flip()
        
        # Wait for user to continue
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                    waiting = False


    def check_button_click(self, mouse_pos):
        controls_y = rows * self.cell_len + PANEL_HEIGHT
        button_width = 100
        button_height = 40
        spacing = 10
        start_x = (self.width - (5*button_width + 4*spacing)) // 2
        
        controls = [
            ("BFS", pygame.K_b),
            ("DFS", pygame.K_d),
            ("A*", pygame.K_a),
            ("Compare", pygame.K_c),
            ("Reset", pygame.K_r)
        ]
        
        for i, (text, key) in enumerate(controls):
            rect = pygame.Rect(
                start_x + i*(button_width + spacing),
                controls_y + (CONTROLS_HEIGHT - button_height)//2,
                button_width,
                button_height
            )
            
            if rect.collidepoint(mouse_pos):
                if key == pygame.K_b:
                    self.run_algorithm("BFS")
                elif key == pygame.K_d:
                    self.run_algorithm("DFS")
                elif key == pygame.K_a:
                    self.run_algorithm("A*")
                elif key == pygame.K_c:
                    self.compare_algorithms()
                elif key == pygame.K_r:
                    self.reset_state()
                return True
        return False

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.VIDEORESIZE:
                    self.handle_resize(event.w, event.h)
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_b:
                        self.path, self.time_taken, self.visited_count = self.bfs(self.start, self.goal, True)
                        self.algorithm = "BFS"
                    elif event.key == pygame.K_d:
                        self.path, self.time_taken, self.visited_count = self.dfs(self.start, self.goal, True)
                        self.algorithm = "DFS"
                    elif event.key == pygame.K_a:
                        self.path, self.time_taken, self.visited_count = self.a_star(self.start, self.goal, True)
                        self.algorithm = "A*"
                    elif event.key == pygame.K_c:
                        self.compare_algorithms()
                    elif event.key == pygame.K_r:
                        self.reset_state()
            
            self.screen.fill(WHITE)
            self.draw_grid()
            
            if self.algorithm:
                self.draw_visited()
                self.draw_path()
            
            self.draw_info_panel()  # This now includes controls
            pygame.display.flip()

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    app = PathfinderApp()
    app.run()