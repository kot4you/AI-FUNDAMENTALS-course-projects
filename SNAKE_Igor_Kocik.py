#!/usr/bin/env python3
from typing import List, Set
from dataclasses import dataclass
import pygame
from enum import Enum, unique
import sys
import random


FPS = 10

INIT_LENGTH = 4

WIDTH = 480
HEIGHT = 480
GRID_SIDE = 24
GRID_WIDTH = WIDTH // GRID_SIDE
GRID_HEIGHT = HEIGHT // GRID_SIDE

BRIGHT_BG = (103, 223, 235)
DARK_BG = (78, 165, 173)

SNAKE_COL = (200, 38, 250)
FOOD_COL = (224, 160, 38)
OBSTACLE_COL = (209, 59, 59)
VISITED_COL = (24, 42, 142)


@unique
class Direction(tuple, Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

    def reverse(self):
        x, y = self.value
        return Direction((x * -1, y * -1))


@dataclass
class Position:
    x: int
    y: int

    def check_bounds(self, width: int, height: int):
        return (self.x >= width) or (self.x < 0) or (self.y >= height) or (self.y < 0)

    def draw_node(self, surface: pygame.Surface, color: tuple, background: tuple):
        r = pygame.Rect(
            (int(self.x * GRID_SIDE), int(self.y * GRID_SIDE)), (GRID_SIDE, GRID_SIDE)
        )
        pygame.draw.rect(surface, color, r)
        pygame.draw.rect(surface, background, r, 1)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Position):
            return (self.x == o.x) and (self.y == o.y)
        else:
            return False

    def __str__(self):
        return f"X{self.x};Y{self.y};"

    def __hash__(self):
        return hash(str(self))


class GameNode:
    nodes: Set[Position] = set()

    def __init__(self):
        self.position = Position(0, 0)
        self.color = (0, 0, 0)

    def randomize_position(self):
        try:
            GameNode.nodes.remove(self.position)
        except KeyError:
            pass

        condidate_position = Position(
            random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1),
        )

        if condidate_position not in GameNode.nodes:
            self.position = condidate_position
            GameNode.nodes.add(self.position)
        else:
            self.randomize_position()

    def draw(self, surface: pygame.Surface):
        self.position.draw_node(surface, self.color, BRIGHT_BG)


class Food(GameNode):
    def __init__(self):
        super(Food, self).__init__()
        self.color = FOOD_COL
        self.randomize_position()


class Obstacle(GameNode):
    def __init__(self):
        super(Obstacle, self).__init__()
        self.color = OBSTACLE_COL
        self.randomize_position()


class Snake:
    def __init__(self, screen_width, screen_height, init_length):
        self.color = SNAKE_COL
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.init_length = init_length
        self.reset()

    def reset(self):
        self.length = self.init_length
        self.positions = [Position((GRID_SIDE // 2), (GRID_SIDE // 2))]
        self.direction = random.choice([e for e in Direction])
        self.score = 0
        self.hasReset = True

    def get_head_position(self) -> Position:
        return self.positions[0]

    def turn(self, direction: Direction):
        if self.length > 1 and direction.reverse() == self.direction:
            return
        else:
            self.direction = direction

    def move(self):
        self.hasReset = False
        cur = self.get_head_position()
        x, y = self.direction.value
        new = Position(cur.x + x, cur.y + y,)
        if self.collide(new):
            self.reset()
        else:
            self.positions.insert(0, new)
            while len(self.positions) > self.length:
                self.positions.pop()

    def collide(self, new: Position):
        return (new in self.positions) or (new.check_bounds(GRID_WIDTH, GRID_HEIGHT))

    def eat(self, food: Food):
        if self.get_head_position() == food.position:
            self.length += 1
            self.score += 1
            while food.position in self.positions:
                food.randomize_position()

    def hit_obstacle(self, obstacle: Obstacle):
        if self.get_head_position() == obstacle.position:
            self.length -= 1
            self.score -= 1
            if self.length == 0:
                self.reset()

    def draw(self, surface: pygame.Surface):
        for p in self.positions:
            p.draw_node(surface, self.color, BRIGHT_BG)


class Player:
    def __init__(self) -> None:
        self.visited_color = VISITED_COL
        self.visited: Set[Position] = set()
        self.chosen_path: List[Direction] = []

    def move(self, snake: Snake) -> bool:
        """
        Try to apply the next planned move.
        Returns:
        - False if a move was successfully applied,
        - True if there was no planned move (path is empty).
        """
        try:
            next_step = self.chosen_path.pop(0)
            snake.turn(next_step)
            return False
        except IndexError:
            return True

    def search_path(self, snake: Snake, food: Food, *obstacles: Set[Obstacle]):
        """
        To be implemented in derived classes.
        """
        pass

    def turn(self, direction: Direction):
        """
        To be implemented in derived classes.
        """
        pass

    def draw_visited(self, surface: pygame.Surface):
        """
        Draw all visited nodes as blue tiles.
        """
        for p in self.visited:
            p.draw_node(surface, self.visited_color, BRIGHT_BG)


class SnakeGame:
    def __init__(self, snake: Snake, player: Player) -> None:
        pygame.init()
        pygame.display.set_caption("AIFundamentals - SnakeGame")

        self.snake = snake
        self.food = Food()
        self.obstacles: Set[Obstacle] = set()
        for _ in range(40):
            ob = Obstacle()
            while any([ob.position == o.position for o in self.obstacles]):
                ob.randomize_position()
            self.obstacles.add(ob)

        self.player = player

        self.fps_clock = pygame.time.Clock()

        self.screen = pygame.display.set_mode(
            (snake.screen_height, snake.screen_width), 0, 32
        )
        self.surface = pygame.Surface(self.screen.get_size()).convert()
        self.myfont = pygame.font.SysFont("monospace", 16)

    def drawGrid(self):
        """
        Draw the chequered background grid.
        """
        for y in range(0, int(GRID_HEIGHT)):
            for x in range(0, int(GRID_WIDTH)):
                p = Position(x, y)
                if (x + y) % 2 == 0:
                    p.draw_node(self.surface, BRIGHT_BG, BRIGHT_BG)
                else:
                    p.draw_node(self.surface, DARK_BG, DARK_BG)

    def run(self):
        while not self.handle_events():
            self.fps_clock.tick(FPS)
            self.drawGrid()

            # Ask the player to set the next direction if:
            # - there is no more planned path (player.move returns True), or
            # - the snake has just been reset.
            if self.player.move(self.snake) or self.snake.hasReset:
                self.player.search_path(self.snake, self.food, self.obstacles)
                self.player.move(self.snake)

            # Move snake according to its direction
            self.snake.move()
            self.snake.eat(self.food)

            # Apply obstacles effects
            for ob in self.obstacles:
                self.snake.hit_obstacle(ob)

            # Draw everything
            for ob in self.obstacles:
                ob.draw(self.surface)
            self.player.draw_visited(self.surface)
            self.snake.draw(self.surface)
            self.food.draw(self.surface)

            self.screen.blit(self.surface, (0, 0))
            text = self.myfont.render(
                "Score {0}".format(self.snake.score), 1, (0, 0, 0)
            )
            self.screen.blit(text, (5, 10))
            pygame.display.update()

    def handle_events(self):
        """
        Handle quit / escape and key presses.
        Arrow keys are forwarded to the player.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_UP:
                    self.player.turn(Direction.UP)
                elif event.key == pygame.K_DOWN:
                    self.player.turn(Direction.DOWN)
                elif event.key == pygame.K_LEFT:
                    self.player.turn(Direction.LEFT)
                elif event.key == pygame.K_RIGHT:
                    self.player.turn(Direction.RIGHT)
        return False


class HumanPlayer(Player):
    def __init__(self):
        super(HumanPlayer, self).__init__()

    def turn(self, direction: Direction):
        """
        For a human player, arrow keys directly append directions to the path.
        """
        self.chosen_path.append(direction)


# ----------------------------------
# DO NOT MODIFY CODE ABOVE THIS LINE
# ----------------------------------

from collections import deque
from typing import Dict, Optional, Tuple
import heapq
from itertools import count


class SearchBasedPlayer(Player):
    """
    AI player that uses graph search to plan a full path to the food.
    Supports:
    - BFS       (unweighted, avoids obstacles if possible)
    - DFS       (unweighted, avoids obstacles if possible)
    - Dijkstra  (weighted, obstacles are expensive)
    - A*        (weighted + heuristic, obstacles are expensive)
    """

    def __init__(self, algorithm: str = "astar"):
        super(SearchBasedPlayer, self).__init__()
        # algorithm ∈ {"bfs", "dfs", "dijkstra", "astar"}
        self.algorithm = algorithm.lower()
        # Counter used to break ties in the priority queue (heapq)
        self._counter = count()

    # ============================================================
    # OVERRIDE DRAWING OF VISITED NODES 
    # ============================================================
    def draw_visited(self, surface: pygame.Surface):
        """
        Draw visited nodes as blue outlines, so that the underlying tile
        (e.g. red obstacle) is still visible.
        """
        for p in self.visited:
            r = pygame.Rect(
                int(p.x * GRID_SIDE),
                int(p.y * GRID_SIDE),
                GRID_SIDE,
                GRID_SIDE,
            )
            # width=2 -> only outline, does not cover the tile color
            pygame.draw.rect(surface, self.visited_color, r, 2)

    # ============================================================
    # MAIN ENTRY POINT – CALLED WHEN PATH IS EMPTY OR SNAKE RESET
    # ============================================================
    def search_path(self, snake: Snake, food: Food, *obstacles: Set[Obstacle]):
        # Always try to search a new path when called.

        # Clear previous path and visited nodes
        self.chosen_path = []
        self.visited.clear()

        # Unpack obstacles set (SnakeGame passes self.obstacles as a single argument)
        obstacle_set: Set[Obstacle] = obstacles[0] if len(obstacles) > 0 else set()
        obstacle_positions = {o.position for o in obstacle_set}

        start: Position = snake.get_head_position()
        goal: Position = food.position
        snake_body_positions = set(snake.positions)

        # Select algorithm
        if self.algorithm == "bfs":
            path_positions = self._bfs(
                start, goal, snake_body_positions, obstacle_positions
            )
        elif self.algorithm == "dfs":
            path_positions = self._dfs(
                start, goal, snake_body_positions, obstacle_positions
            )
        elif self.algorithm == "dijkstra":
            path_positions = self._dijkstra(
                start, goal, snake_body_positions, obstacle_positions
            )
        elif self.algorithm == "astar":
            path_positions = self._a_star(
                start, goal, snake_body_positions, obstacle_positions
            )
        else:
            path_positions = []

        # If no path found – just don't change direction (snake keeps last one)
        if not path_positions:
            return

        # Convert list of positions to list of Directions
        self.chosen_path = self._positions_to_directions(path_positions)

    # ============================================================
    # CHANGE ALGORITHM WITH ARROW KEYS
    # ============================================================
    def turn(self, direction: Direction):
        """
        Map arrow keys to algorithms:
        - UP    -> BFS
        - DOWN  -> DFS
        - LEFT  -> Dijkstra
        - RIGHT -> A*
        """
        if direction == Direction.UP:
            self.algorithm = "bfs"
        elif direction == Direction.DOWN:
            self.algorithm = "dfs"
        elif direction == Direction.LEFT:
            self.algorithm = "dijkstra"
        elif direction == Direction.RIGHT:
            self.algorithm = "astar"

        # When algorithm changes, clear current path
        self.chosen_path = []

    # ============================================================
    # NEIGHBOR GENERATION HELPERS
    # ============================================================
    def _get_neighbors_blocked(
        self,
        current: Position,
        snake_body: Set[Position],
        blocked: Set[Position],
    ) -> List[Position]:
        """
        Generate valid neighboring positions (up, down, left, right),
        ignoring:
        - cells outside the board,
        - cells occupied by the snake body,
        - cells in 'blocked' set (e.g. obstacles).
        """
        neighbors: List[Position] = []
        for d in Direction:
            dx, dy = d.value
            new = Position(current.x + dx, current.y + dy)

            # Outside bounds?
            if new.check_bounds(GRID_WIDTH, GRID_HEIGHT):
                continue

            # Snake body treated as obstacles
            if new in snake_body:
                continue

            # Extra blocked tiles (e.g. obstacles)
            if new in blocked:
                continue

            neighbors.append(new)
        return neighbors

    def _get_neighbors_basic(
        self, current: Position, snake_body: Set[Position]
    ) -> List[Position]:
        """
        Neighbors ignoring obstacles (only walls and snake body).
        Used by Dijkstra / A* which handle obstacles via cost.
        """
        return self._get_neighbors_blocked(current, snake_body, blocked=set())

    # ============================================================
    # BFS – TWO-PHASE: FIRST AVOID OBSTACLES, THEN ALLOW THEM
    # ============================================================
    def _bfs_core(
        self,
        start: Position,
        goal: Position,
        snake_body: Set[Position],
        blocked: Set[Position],
    ) -> Tuple[List[Position], Set[Position]]:
        """
        Single BFS run with a given 'blocked' set.
        Returns:
        - (path, visited_set)
        """
        queue: deque[Position] = deque()
        queue.append(start)

        came_from: Dict[Position, Optional[Position]] = {start: None}
        visited_local: Set[Position] = {start}

        while queue:
            current = queue.popleft()

            if current == goal:
                break

            for neighbor in self._get_neighbors_blocked(current, snake_body, blocked):
                if neighbor not in came_from:
                    came_from[neighbor] = current
                    queue.append(neighbor)
                    visited_local.add(neighbor)

        if goal not in came_from:
            return [], visited_local

        path = self._reconstruct_path(came_from, start, goal)
        return path, visited_local

    def _bfs(
        self,
        start: Position,
        goal: Position,
        snake_body: Set[Position],
        obstacle_positions: Set[Position],
    ) -> List[Position]:
        """
        BFS:
        1. First tries to find a path that does NOT step on obstacles
           (obstacles treated as walls).
        2. If that fails, runs a second BFS that allows stepping on obstacles.
        Visited nodes from both attempts are drawn in blue.
        """
        # Phase 1: obstacles blocked
        path1, visited1 = self._bfs_core(
            start, goal, snake_body, blocked=obstacle_positions
        )
        self.visited |= visited1

        if path1:
            return path1

        # Phase 2: obstacles allowed
        path2, visited2 = self._bfs_core(
            start, goal, snake_body, blocked=set()
        )
        self.visited |= visited2

        return path2  # may be [] if no path at all

    # ============================================================
    # DFS – TWO-PHASE (AVOID THEN ALLOW OBSTACLES)
    # ============================================================
    def _dfs_core(
        self,
        start: Position,
        goal: Position,
        snake_body: Set[Position],
        blocked: Set[Position],
    ) -> Tuple[List[Position], Set[Position]]:
        """
        Single DFS run with a given 'blocked' set.
        Returns:
        - (path, visited_set)
        """
        stack: List[Position] = [start]
        came_from: Dict[Position, Optional[Position]] = {start: None}
        visited_local: Set[Position] = {start}

        while stack:
            current = stack.pop()

            if current == goal:
                break

            # For DFS we don't need any special ordering; blocked controls obstacles
            for neighbor in self._get_neighbors_blocked(current, snake_body, blocked):
                if neighbor not in came_from:
                    came_from[neighbor] = current
                    stack.append(neighbor)
                    visited_local.add(neighbor)

        if goal not in came_from:
            return [], visited_local

        path = self._reconstruct_path(came_from, start, goal)
        return path, visited_local

    def _dfs(
        self,
        start: Position,
        goal: Position,
        snake_body: Set[Position],
        obstacle_positions: Set[Position],
    ) -> List[Position]:
        """
        DFS:
        1. First tries to find a path that does NOT step on obstacles.
        2. If that fails, runs a second DFS that allows stepping on obstacles.
        """
        # Phase 1: obstacles blocked
        path1, visited1 = self._dfs_core(
            start, goal, snake_body, blocked=obstacle_positions
        )
        self.visited |= visited1

        if path1:
            return path1

        # Phase 2: obstacles allowed
        path2, visited2 = self._dfs_core(
            start, goal, snake_body, blocked=set()
        )
        self.visited |= visited2

        return path2

    # ============================================================
    # DIJKSTRA – SHORTEST PATH WITH POSITIVE EDGE COSTS
    # ============================================================
    def _dijkstra(
        self,
        start: Position,
        goal: Position,
        snake_body: Set[Position],
        obstacle_positions: Set[Position],
    ) -> List[Position]:
        """
        Dijkstra's algorithm:
        - Uses a priority queue (min-heap) ordered by total cost from the start.
        - Movement cost is defined by _movement_cost() and depends on obstacles.
        Obstacles are *not* blocked here, just more expensive.
        """

        # Priority queue elements are tuples: (cost_from_start, counter, position)
        frontier: List[Tuple[int, int, Position]] = []
        heapq.heappush(frontier, (0, next(self._counter), start))

        came_from: Dict[Position, Optional[Position]] = {start: None}
        cost_so_far: Dict[Position, int] = {start: 0}
        self.visited.add(start)

        while frontier:
            current_cost, _, current = heapq.heappop(frontier)

            if current == goal:
                break

            for neighbor in self._get_neighbors_basic(current, snake_body):
                # Cost of moving into the neighbor
                move_cost = self._movement_cost(neighbor, obstacle_positions)
                new_cost = current_cost + move_cost

                # Relaxation step
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    came_from[neighbor] = current
                    # Priority for Dijkstra is simply the new total cost
                    priority = new_cost
                    heapq.heappush(
                        frontier, (priority, next(self._counter), neighbor)
                    )
                    self.visited.add(neighbor)

        if goal not in came_from:
            # No path found
            return []

        return self._reconstruct_path(came_from, start, goal)

    # ============================================================
    # A* – DIJKSTRA + HEURISTIC
    # ============================================================
    def _a_star(
        self,
        start: Position,
        goal: Position,
        snake_body: Set[Position],
        obstacle_positions: Set[Position],
    ) -> List[Position]:
        """
        A* search:
        - Uses f(n) = g(n) + h(n),
          where g(n) is the cost from the start,
          and h(n) is the heuristic estimate to the goal (Manhattan distance).
        Obstacles are not blocked, just more expensive.
        """

        # Priority queue elements: (priority_f, counter, position)
        frontier: List[Tuple[int, int, Position]] = []
        heapq.heappush(frontier, (0, next(self._counter), start))

        came_from: Dict[Position, Optional[Position]] = {start: None}
        cost_so_far: Dict[Position, int] = {start: 0}
        self.visited.add(start)

        while frontier:
            current_priority, _, current = heapq.heappop(frontier)

            if current == goal:
                break

            for neighbor in self._get_neighbors_basic(current, snake_body):
                # Cost of moving into neighbor
                move_cost = self._movement_cost(neighbor, obstacle_positions)
                new_cost = cost_so_far[current] + move_cost

                # Relaxation with A* (using f = g + h)
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    came_from[neighbor] = current
                    priority = new_cost + self._heuristic(neighbor, goal)
                    heapq.heappush(
                        frontier, (priority, next(self._counter), neighbor)
                    )
                    self.visited.add(neighbor)

        if goal not in came_from:
            # No path found
            return []

        return self._reconstruct_path(came_from, start, goal)

    # ============================================================
    # HELPER FUNCTIONS
    # ============================================================
    def _movement_cost(
        self, pos: Position, obstacle_positions: Set[Position]
    ) -> int:
        """
        Cost function for Dijkstra/A*:
        - normal tile: cost 1
        - obstacle tile (red): higher cost, e.g. 10
        This makes the algorithms avoid obstacles if possible,
        but still allows stepping on them when necessary.
        """
        if pos in obstacle_positions:
            return 10
        return 1

    def _heuristic(self, a: Position, b: Position) -> int:
        """
        Heuristic function for A*.
        Manhattan distance works well on a 4-connected grid.
        """
        return abs(a.x - b.x) + abs(a.y - b.y)

    def _reconstruct_path(
        self,
        came_from: Dict[Position, Optional[Position]],
        start: Position,
        goal: Position,
    ) -> List[Position]:
        """
        Reconstructs path from start to goal using the came_from dictionary.
        The result is a list of Positions from start to goal (inclusive).
        """
        path: List[Position] = []
        current = goal
        while current is not None and current != start:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()
        return path

    def _positions_to_directions(self, path: List[Position]) -> List[Direction]:
        """
        Converts a list of consecutive positions [p0, p1, ..., pk]
        into a list of Directions [d1, d2, ..., dk],
        where d_i is the move from p_{i-1} to p_i.
        """
        directions: List[Direction] = []
        if len(path) < 2:
            return directions

        for prev, nxt in zip(path[:-1], path[1:]):
            dx = nxt.x - prev.x
            dy = nxt.y - prev.y
            if (dx, dy) == Direction.UP.value:
                directions.append(Direction.UP)
            elif (dx, dy) == Direction.DOWN.value:
                directions.append(Direction.DOWN)
            elif (dx, dy) == Direction.LEFT.value:
                directions.append(Direction.LEFT)
            elif (dx, dy) == Direction.RIGHT.value:
                directions.append(Direction.RIGHT)

        return directions


if __name__ == "__main__":
    snake = Snake(WIDTH, WIDTH, INIT_LENGTH)

    # Default algorithm is A*, but you can also use: "bfs", "dfs" or "dijkstra"
    algorithm = "astar"
    if len(sys.argv) >= 2:
        algorithm = sys.argv[1].lower()

    # Uncomment for manual control:
    # player = HumanPlayer()

    # AI player:
    player = SearchBasedPlayer(algorithm=algorithm)

    game = SnakeGame(snake, player)
    game.run()
