import numpy as np
import torch
import os
import pickle
from datetime import datetime

from src.engine.board import HexBoard, Player
from src.agent.mcts import MCTS
from src.agent.model import HexAlphaNet

class SelfPlayWorker:
    """Handles the generation of self-play games to create RL training data."""
    
    def __init__(self, model_path=None, num_simulations=400, board_size=11):
        self.board_size = board_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the Brain
        self.model = HexAlphaNet(board_size=self.board_size).to(self.device)
        self.model.eval() # Must be in eval mode for MCTS inference!
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model weights from {model_path}")
            
        self.mcts = MCTS(self.model, num_simulations=num_simulations)

    def play_game(self, temperature=1.0):
        """Simulates a single game of Hex and returns the training data."""
        board = HexBoard(size=self.board_size)
        current_player = Player.RED
        
        # Store states, MCTS policies, and current player for backfilling rewards
        states = []
        policies = []
        players = []
        
        while True:
            # 1. Get the improved policy from MCTS
            # We use temperature=1.0 early on to ensure the AI explores different openings
            mcts_policy = self.mcts.get_action_probs(board, current_player, temperature=temperature)
            
            # 2. Store the current state and policy
            state_tensor = self.mcts._board_to_tensor(board, current_player).cpu().numpy()
            states.append(state_tensor)
            policies.append(mcts_policy)
            players.append(current_player)
            
            # 3. Choose a move probabilistically based on the MCTS policy
            action = np.random.choice(len(mcts_policy), p=mcts_policy)
            row, col = divmod(action, self.board_size)
            
            board.place_piece(row, col, current_player)
            
            # 4. Check for a win
            if board.check_win(current_player):
                winner = current_player
                break
                
            # Swap turns
            current_player = Player.BLUE if current_player == Player.RED else Player.RED

        # 5. Backfill the Rewards (Z)
        # If the player for that turn is the winner, Z = 1. Else Z = -1.
        rewards = [1.0 if p == winner else -1.0 for p in players]
        
        return states, policies, rewards

    def generate_batch(self, num_games=10, save_dir="training/data"):
        """Plays multiple games and saves the dataset for the training loop."""
        os.makedirs(save_dir, exist_ok=True)
        
        all_states, all_policies, all_rewards = [], [], []
        
        print(f"Starting generation of {num_games} self-play games...")
        for i in range(num_games):
            s, p, r = self.play_game(temperature=1.0)
            all_states.extend(s)
            all_policies.extend(p)
            all_rewards.extend(r)
            print(f"Game {i+1}/{num_games} completed. ({len(s)} moves)")
            
        # Package and save
        dataset = {
            "states": np.array(all_states, dtype=np.float32),
            "policies": np.array(all_policies, dtype=np.float32),
            "rewards": np.array(all_rewards, dtype=np.float32)
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(save_dir, f"batch_{timestamp}.pkl")
        
        with open(filepath, "wb") as f:
            pickle.dump(dataset, f)
            
        print(f"Saved {len(all_states)} total positions to {filepath}")

if __name__ == "__main__":
    # real dataset generation run
    worker = SelfPlayWorker(board_size=11, num_simulations=400)
    worker.generate_batch(num_games=100)