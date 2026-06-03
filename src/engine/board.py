import numpy as np
from enum import IntEnum

class Player(IntEnum):
    """
    Using integers makes the board highly optimized for Machine Learning tensors.
    0 = Empty, 1 = Red, -1 = Blue
    """
    EMPTY = 0
    RED = 1
    BLUE = -1

class HexBoard:
    def __init__(self, size=5):
        self.size = size
        # Initialize a matrix of zeros (Empty)
        self.grid = np.full((size, size), Player.EMPTY, dtype=int)
        
    def is_valid_move(self, row, col):
        """Checks if a coordinate is on the board and currently empty."""
        if 0 <= row < self.size and 0 <= col < self.size:
            return self.grid[row, col] == Player.EMPTY
        return False

    def place_piece(self, row, col, player):
        """Places a piece and returns True if successful, False otherwise."""
        if self.is_valid_move(row, col):
            self.grid[row, col] = player
            return True
        raise ValueError(f"Invalid move at ({row}, {col}).")

    def get_neighbors(self, row, col):
        """Returns a list of valid (on-board) neighboring coordinates."""
        # The 6 directions in a skewed Hex grid
        directions = [
            (-1, 0), (-1, 1),  # Top-left, Top-right
            (0, -1), (0, 1),   # Left, Right
            (1, -1), (1, 0)    # Bottom-left, Bottom-right
        ]
        
        neighbors = []
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < self.size and 0 <= c < self.size:
                neighbors.append((r, c))
                
        return neighbors

    def display_terminal(self):
        """A quick and dirty printer just to test our logic in the console."""
        symbols = {Player.EMPTY: ".", Player.RED: "R", Player.BLUE: "B"}
        for r in range(self.size):
            # Add spaces to visually skew the printed matrix into a rhombus
            indent = " " * r
            row_str = " ".join([symbols[Player(val)] for val in self.grid[r]])
            print(f"{indent}{row_str}")
    
def check_win(self, player):
        """Uses Depth-First Search (DFS) to check if a player has connected their edges."""
        # 1. Define the starting edges and the target edges
        if player == Player.RED:
            # Red starts at the Top (row 0) and wants to reach the Bottom
            start_nodes = [(0, c) for c in range(self.size) if self.grid[0, c] == player]
            def is_target(r, c): return r == self.size - 1
            
        elif player == Player.BLUE:
            # Blue starts at the Left (col 0) and wants to reach the Right
            start_nodes = [(r, 0) for r in range(self.size) if self.grid[r, 0] == player]
            def is_target(r, c): return c == self.size - 1
            
        else:
            return False

        # 2. Run the Depth-First Search
        visited = set(start_nodes)
        stack = list(start_nodes)

        while stack:
            r, c = stack.pop()
            
            # If our current hex touches the target edge, it's a guaranteed win!
            if is_target(r, c):
                return True
                
            # Check all valid, on-board neighbors
            for nr, nc in self.get_neighbors(r, c):
                # If the neighbor belongs to the same player and hasn't been checked yet
                if (nr, nc) not in visited and self.grid[nr, nc] == player:
                    visited.add((nr, nc))
                    stack.append((nr, nc))
                    
        # If the stack empties and we never hit the target edge, no win yet
        return False