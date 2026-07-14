import math
import copy
import numpy as np
import torch
import torch.nn.functional as F

from src.engine.board import Player

class Node:
    """A node in the MCTS tree representing a specific board state."""
    def __init__(self, prior_prob, parent=None):
        self.prior_prob = prior_prob  # P(s, a)
        self.visit_count = 0          # N(s, a)
        self.value_sum = 0.0          # W(s, a)
        self.children = {}            # Action index -> Node
        self.parent = parent          # Parent Node

    @property
    def q_value(self):
        """Returns the mean action value Q(s, a)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

class MCTS:
    """Monte Carlo Tree Search driven by the Dual-Headed ResNet."""
    def __init__(self, model, c_puct=1.5, num_simulations=400):
        self.model = model
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.device = next(model.parameters()).device

    def get_action_probs(self, board, current_player, temperature=1.0):
        """Runs the MCTS simulations and returns the move probability distribution."""
        root = Node(prior_prob=1.0)
        
        # Add Dirichlet noise for root exploration in self-play
        # (This prevents the AI from always playing the exact same determinisic game)
        
        for _ in range(self.num_simulations):
            # 1. Selection
            node = root
            sim_board = copy.deepcopy(board) # Isolate state for forward simulation
            sim_player = current_player
            
            while node.children:
                action, node = self._select_child(node)
                row, col = divmod(action, sim_board.size)
                sim_board.place_piece(row, col, sim_player)
                
                # Check terminal state
                if sim_board.check_win(sim_player):
                    break
                sim_player = Player.BLUE if sim_player == Player.RED else Player.RED

            # 2. Expansion and Evaluation
            # (If it's not a terminal state, evaluate with NN and expand)
            value = -1.0 # Default to win for current player if terminal
            if not sim_board.check_win(sim_player):
                policy_logits, value_tensor = self._evaluate_board(sim_board, sim_player)
                value = value_tensor.item()
                
                # Mask illegal moves (already placed pieces)
                valid_moves = [r * sim_board.size + c 
                               for r in range(sim_board.size) for c in range(sim_board.size) 
                               if sim_board.grid[r, c] == Player.EMPTY]
                
                policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
                policy_sum = sum(policy[a] for a in valid_moves)
                
                # Expand node with valid moves
                for action in valid_moves:
                    # Renormalize priors after masking
                    prior = policy[action] / policy_sum if policy_sum > 0 else 1.0 / len(valid_moves)
                    node.children[action] = Node(prior_prob=prior, parent=node)

            # 3. Backpropagation
            # Walk back up the tree and update W and N
            self._backpropagate(node, value)

        # 4. Action Selection via Temperature
        action_visits = {a: child.visit_count for a, child in root.children.items()}
        actions = list(action_visits.keys())
        counts = list(action_visits.values())

        if temperature == 0:
            # Greedy play (for tournament/inference)
            best_action = actions[np.argmax(counts)]
            probs = np.zeros(board.size * board.size)
            probs[best_action] = 1.0
            return probs

        # Exploratory play (for self-play training)
        counts = np.array(counts) ** (1.0 / temperature)
        probs = counts / np.sum(counts)
        
        full_probs = np.zeros(board.size * board.size)
        for a, p in zip(actions, probs):
            full_probs[a] = p
            
        return full_probs

    def _select_child(self, node):
        """Selects the child with the highest PUCT score."""
        best_score = -float('inf')
        best_action = -1
        best_child = None

        for action, child in node.children.items():
            u = self.c_puct * child.prior_prob * math.sqrt(node.visit_count) / (1 + child.visit_count)
            # We invert the child's Q-value because Hex is zero-sum (my loss is your win)
            score = -child.q_value + u 
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        return best_action, best_child

    def _backpropagate(self, node, value):
        """Updates visit counts and values up the tree."""
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            # Flip perspective for the parent node
            value = -value 
            node = node.parent

    @torch.no_grad()
    def _evaluate_board(self, board, current_player):
        """Compiles the board into a tensor and runs network inference."""
        tensor = self._board_to_tensor(board, current_player).unsqueeze(0).to(self.device)
        return self.model(tensor)

    def _board_to_tensor(self, board, current_player):
        """Converts the HexBoard into a (3, 11, 11) binary plane tensor."""
        size = board.size
        tensor = torch.zeros(3, size, size, dtype=torch.float32)
        
        opponent = Player.BLUE if current_player == Player.RED else Player.RED
        
        # Plane 0: Current Player's pieces
        tensor[0][board.grid == current_player] = 1.0
        # Plane 1: Opponent's pieces
        tensor[1][board.grid == opponent] = 1.0
        # Plane 2: Turn indicator (1.0 if Red, 0.0 if Blue)
        if current_player == Player.RED:
            tensor[2].fill_(1.0)
            
        return tensor