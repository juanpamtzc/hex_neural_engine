import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import numpy as np
import time
import onnxruntime as ort
import json
import os

from src.engine.board import HexBoard, Player

# AI Inference setup using ONNX
@st.cache_resource
def load_ai_model(model_path):
    """loads a specific ONNX model for AI inference, cached by file path"""
    try:
        if not model_path or not os.path.exists(model_path):
            return None
        return ort.InferenceSession(model_path)
    except Exception as e:
        st.error(f"Failed to load AI model at {model_path}: {e}")
        return None
    
def get_availabl_models():
    """Scans models directory and returns sorted dict of available models"""
    models = {"Random Baseline (Untrained)": None}

    # add production model if it exists
    if os.path.exists("models/production/hex_model.onnx"):
        models["Latest Production"] = "models/production/hex_model.onnx"

    # Scan archive folder for specific generations
    archive_dir = "models/archive"
    if os.path.exists(archive_dir):
        # Sort files so more developed models appear above
        for f in sorted(os.listdir(archive_dir), key=lambda x: [int(s) for s in x.split('_') if s.isdigit()] of [0], reverse=True):
            if f.endswith(".onnx"):
                gen_num = "".join(filter(str.isdigit, f))
                models[f"Generation {gen_num}"] = os.path.join(archive_dir, f)
    
    return models

# Load the AI model once at startup
ai_session = load_ai_model()

def load_model_metadata():
    """Loads runtime metadata about the current model generation."""
    try:
        with open("models/production/metadata.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # Default fallback for the current initial bootstrap model
        return {
            "generation": 1,
            "total_games": 100,
            "status": "Bootstrap Phase (Pure Exploration)",
            "architecture": "AlphaZero ResNet Dual-Head"
        }

def board_to_numpy_tensor(board, current_player):
    """Compiles the board into a (1, 3, board_size, board_size) float32 tensor suitable for ONNX inference"""
    tensor = np.zeros((1, 3, board.size, board.size), dtype=np.float32)
    opponent = Player.BLUE if current_player == Player.RED else Player.RED

    # plane 0: current player pieces
    tensor[0, 0][board.grid == current_player] = 1.0
    # plane 1: opponent pieces
    tensor[0, 1][board.grid == opponent] = 1.0
    # plane 2: turn indicator
    if current_player == Player.RED:
        tensor[0, 2].fill(1.0)
    
    return tensor

def get_ai_move(board, player, session):
    """Passes the board state through the ONNX static graph"""
    if session is None:
        return None, None
    
    valid_moves = [(r, c) for r in range(board.size) for c in range(board.size) if board.grid[r,c] == Player.EMPTY]
    if not valid_moves:
        return None, None

    input_tensor = board_to_numpy_tensor(board, player)
    ort_inputs = {session.get_inputs()[0].name: input_tensor}
    policy_logits, _ = session.run(None, ort_inputs)

    policy = policy_logits[0]
    masked_policy = np.full(board.size * board.size, -np.inf)

    valid_indices = [r * board.size + c for r, c in valid_moves]
    for idx in valid_indices:
        masked_policy[idx] = policy[idx]
    
    best_action = np.argmax(masked_policy)
    best_row, best_col = divmod(best_action, board.size)
    
    time.sleep(0.3) # delay for UI smoothness
    return best_row, best_col

def get_policy_heatmap(board, player, session):
    """Runs the board through the AI and returns an 11x11 grid of probabilities."""
    if session is None:
        return None
        
    input_tensor = board_to_numpy_tensor(board, player)
    ort_inputs = {session.get_inputs()[0].name: input_tensor}
    policy_logits, _ = session.run(None, ort_inputs)
    
    policy = policy_logits[0]
    
    # Mask out illegal moves with negative infinity
    valid_mask = (board.grid == Player.EMPTY).flatten()
    masked_policy = np.where(valid_mask, policy, -np.inf)
    
    # Apply stable Softmax
    max_logit = np.max(masked_policy)
    if max_logit == -np.inf: # No valid moves left
        return np.zeros((board.size, board.size))
        
    exp_preds = np.exp(masked_policy - max_logit)
    probs = exp_preds / np.sum(exp_preds)
    
    return probs.reshape((board.size, board.size))

# 1. Page Setup
st.set_page_config(page_title="Hex-Zero Arena", layout="centered")
st.title("Hex-Zero: Interactive Arena")
st.markdown("Red connects **Top-Right to Bottom-Left**. Blue connects **Top-Left to Bottom-Right**.")

# 2. Session State Initialization
if 'board' not in st.session_state:
    st.session_state.board = HexBoard(size=11)
    st.session_state.current_player = Player.RED
    st.session_state.winner = None

# 3. Matplotlib Rendering Function
def draw_board(board_obj, heatmap_probs=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    radius = 0.5
    
    color_map = {
        Player.EMPTY: '#f0f2f6', 
        Player.RED: '#ff4b4b',   
        Player.BLUE: '#1f77b4'   
    }
    
    for row in range(board_obj.size):
        for col in range(board_obj.size):
            # The affine projection for a symmetrical diamond lattice
            x = (col - row) * 1.5 * radius
            y = -(col + row) * (np.sqrt(3) / 2 * radius)
            
            player_at_pos = board_obj.grid[row, col]

            hex_color = color_map[player_at_pos]
            if player_at_pos == Player.EMPTY and heatmap_probs is not None:
                prob = heatmap_probs[row, col]
                max_prob = np.max(heatmap_probs) if np.max(heatmap_probs) > 0 else 1
                intensity = prob / max_prob

                # only add color if there's at least a 5% consideration
                if intensity > 0.05:
                    hex_color = plt.cm.plasma(intensity)
            

            hexagon = RegularPolygon(
                (x, y),
                numVertices=6,
                radius=radius,
                orientation=np.radians(30), 
                facecolor=hex_color,
                edgecolor='black',
                linewidth=1.5
            )
            ax.add_patch(hexagon)
            
            text_color = 'white' if player_at_pos != Player.EMPTY else '#888888'
            ax.text(x, y, f"{row},{col}", ha='center', va='center', fontsize=6, color=text_color, fontweight='bold')

    ax.autoscale_view()
    ax.axis('off')
    return fig

# 4. Game UI & Logic
with st.sidebar:
    st.header("Game Settings")

    available_models = get_available_models()
    
    st.subheader("Match Configuration")

    p1_type = st.selectbox("Player 1 (RED):", ("Human", "AI"))
    red_model_path = None
    if p1_type == "AI":
        selected_red = st.selectbox("Select RED model:", list(available_models.keys()), key="red_model")
        red_model_path = available_models[selected_red]
    
    st.write("---")

    p2_type = st.selectbox("Player 2 (BLUE):", ("Human", "AI"))
    blue_model_path = None
    if p2_type == "AI":
        selected_blue = st.selectbox("Select BLUE model:", list(available_models.keys()), key="blue_model")
        blue_model_path = available_models[selected_blue]

    # Fetch latest pipeline metrics
    meta = load_model_metadata()
    
    st.divider()
    st.subheader("Model Intelligence Ledger")
    
    # Clean, metric grid display
    col_g, col_s = st.columns(2)
    with col_g:
        st.metric(label="Latest Gen", value=f"v{meta['generation']}")
    with col_s:
        st.metric(label="Experience Base", value=f"{meta['total_games']} games")
        
    st.caption(f"**Current State:** {meta['status']}")
    
    # Educational expander explaining the PSPACE-complete learning curve
    with st.expander("What to expect from this AI?", expanded=False):
        st.markdown("""
        Hex has a massive state space ($3^{121}$ combinations). Because we are running fast "micro-generations" (100 games each) locally, the learning curve is extended:
        
        **Phase 1: Gen 1–25 (The Bootstrap)**
        *High-entropy noise.* The model understands the rules but lacks tactical sight. It fills the board semi-randomly, learning basic graph topology.
        
        **Phase 2: Gen 26–100 (Emergence)**
        *Basic clustering.* The network discovers that the center hexes hold higher graph centrality. It begins prioritizing the middle and executing simple 1-move blocks.
        
        **Phase 3: Gen 100+ (Theory & Mastery)**
        *Virtual Connections.* The network unlocks Hex game theory natively, intentionally building unblockable **Two-Bridges** and forcing tactical bottlenecks.
        
        *More self-play cycles are actively rendering on the workstation GPU.*
        """)

    st.toggle("Show AI's Strategy (Policy Heatmap)", value=False, key="show_heatmap")
    show_heatmap = st.session_state.show_heatmap
    
    
    st.divider()
    st.header("Match Status")
    
    if st.session_state.winner:
        st.success(f"🏆 {st.session_state.winner.name} WINS!")
        if st.button("Play Again"):
            st.session_state.board = HexBoard(size=st.session_state.board.size)
            st.session_state.winner = None
            st.session_state.current_player = Player.RED
            st.rerun()
    else:
        player_color = "🔴 RED" if st.session_state.current_player == Player.RED else "🔵 BLUE"
        st.markdown(f"**Current Turn:** {player_color}")
        
        # determine if current player is AI and which path to use
        is_ai_turn = False
        current_model_path = None

        if st.session_state.current_player == Player.RED and p1_type == "AI":
            is_ai_turn = True
            current_model_path = red_model_path
        elif st.session_state.current_player == Player.BLUE and p2_type == "AI":
            is_ai_turn = True
            current_model_path = blue_model_path

        # for now, I'm having the AI have it's own button.
        if is_ai_turn:
            if st.button("Generate AI Move", type="primary"):
                active_session = load_ai_model(current_model_path)

                row, col = get_ai_move(st.session_state.board, st.session_state.current_player, active_session)
                if row is not None and col is not None:
                    st.session_state.board.place_piece(row, col, st.session_state.current_player)
                    
                    if st.session_state.board.check_win(st.session_state.current_player):
                        st.session_state.winner = st.session_state.current_player
                    else:
                        st.session_state.current_player = Player.BLUE if st.session_state.current_player == Player.RED else Player.RED
                    st.rerun()
        else:
            col1, col2 = st.columns(2)
            with col1:
                row_input = st.number_input("Row", min_value=0, max_value=st.session_state.board.size-1, value=0)
            with col2:
                col_input = st.number_input("Column", min_value=0, max_value=st.session_state.board.size-1, value=0)
                
            if st.button("Place Piece"):
                if st.session_state.board.is_valid_move(row_input, col_input):
                    st.session_state.board.place_piece(row_input, col_input, st.session_state.current_player)
                    
                    if st.session_state.board.check_win(st.session_state.current_player):
                        st.session_state.winner = st.session_state.current_player
                    else:
                        st.session_state.current_player = Player.BLUE if st.session_state.current_player == Player.RED else Player.RED
                    st.rerun()
                else:
                    st.error("Invalid move! That hex is already taken.")

    

# 5. Draw the current state to the screen
heatmap_data = None
if st.session_state.get("show_heatmap", False) and not st.session_state.winner:
    active_path = red_model_path if st.session_state.current_player == Player.RED else blue_model_path
    active_session = load_ai_model(active_path)
    heatmap_data = get_policy_heatmap(st.session_state.board, st.session_state.current_player, active_session)

st.pyplot(draw_board(st.session_state.board, heatmap_probs=heatmap_data))

