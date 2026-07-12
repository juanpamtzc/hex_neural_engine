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
    def __init__(self, size=11):
        self.size = size
        self.grid = np.full((size, size), Player.EMPTY, dtype=int)
        
        # --- UNION-FIND INITIALIZATION ---
        # We need size*size nodes for the board, plus 4 virtual nodes for the edges.
        total_nodes = (size * size) + 4
        self.parent = np.arange(total_nodes, dtype=int)
        
        # Define virtual edge indices (sitting at the end of the 1D array)
        self.RED_TOP = size * size
        self.RED_BOT = size * size + 1
        self.BLUE_LFT = size * size + 2
        self.BLUE_RGT = size * size + 3
        
    def is_valid_move(self, row, col):
        """Checks if a coordinate is on the board and currently empty."""
        if 0 <= row < self.size and 0 <= col < self.size:
            return self.grid[row, col] == Player.EMPTY
        return False

    def place_piece(self, row, col, player):
        """Places a piece and updates connected components using union-find."""
        if not self.is_valid_move(row,col):
            raise ValueError(f"Invalid move at ({row}, {col})")
        
        self.grid[row, col] = player

        idx = row * self.size + col

        if player == Player.RED:
            if row==0: 
                self._union(idx, self.RED_TOP)
            if row==self.size - 1:
                self._union(idx, self.RED_BOT)
        elif player == Player.BLUE:
            if col==0:
                self._union(idx, self.BLUE_LFT)
            if col==self.size-1:
                self._union(idx, self.BLUE_RGT)
        
        for nr, nc in self.get_neighbors(row,col):
            if self.grid[nr, nc]==player:
                n_idx = nr * self.size + nc
                self._union(idx, n_idx)
        
        return True

    def get_neighbors(self, row, col):
        """Returns a list of valid (on-board) neighboring coordinates."""
        # The 6 directions in a skewed Hex grid
        directions = [
            (0, -1), (-1, 0),  # Top-left, Top-right
            (1, 1), (-1, -1),   # Left, Right
            (1, 0), (0, 1)    # Bottom-left, Bottom-right
        ]
        
        neighbors = []
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < self.size and 0 <= c < self.size:
                neighbors.append((r, c))
                
        return neighbors
    
    # UNION-FIND ALGORITHM FUNCTIONS
    # Since depth-first search would be too expensive, switching to union-find algorithm
    def _find(self, i):
        """Finds the root of node i with path compression for O(1) amortized time."""
        if self.parent[i] == i:
            return i
        self.parent[i] = self._find(self.parent[i])
        return self.parent[i]
    # UNION-FIND ALGORITHM FUNCTIONS
    def _union(self, i, j):
        """Connects two nodes/components."""
        root_i = self._find(i)
        root_j = self._find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j

    def check_win(self, player):
        """O(1) amortized win check using Union-Find."""
        if player == Player.RED:
            return self._find(self.RED_TOP) == self._find(self.RED_BOT)
        elif player == Player.BLUE:
            return self._find(self.BLUE_LFT) == self._find(self.BLUE_RGT)
        return False

    def display_terminal(self):
        """A quick and dirty printer just to test our logic in the console."""
        symbols = {Player.EMPTY: ".", Player.RED: "R", Player.BLUE: "B"}
        for r in range(self.size):
            # Add spaces to visually skew the printed matrix into a rhombus
            indent = " " * r
            row_str = " ".join([symbols[Player(val)] for val in self.grid[r]])
            print(f"{indent}{row_str}")