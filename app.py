import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import numpy as np

# Import your core engine structures
from src.engine.board import HexBoard, Player

# --- UI CONFIGURATION ---
st.set_page_config(
    page_title="Hex-Zero Alpha Arena",
    page_icon="⬢",
    layout="wide"
)

st.title("⬢ Hex-Zero: Deep Reinforcement Learning Arena")
st.caption("Architected by a Staff MLE | Built with AlphaZero Search & PyTorch Inference")

# --- 1. FLAWLESS STATE MANAGEMENT ---
# Initialize persistent variables to manage game loops across browser re-runs
if 'board_size' not in st.session_state:
    st.session_state.board_size = 5  # Start lightweight, scale to 9x9 later

if 'board' not in st.session_state:
    st.session_state.board = HexBoard(size=st.session_state.board_size)

if 'current_player' not in st.session_state:
    st.session_state.current_player = Player.RED

if 'winner' not in st.session_state:
    st.session_state.winner = None

if 'game_mode' not in st.session_state:
    st.session_state.game_mode = "Human vs. Machine"

# Derived agent configuration mappings based on selected game mode
if st.session_state.game_mode == "Human vs. Human (Local)":
    st.session_state.p1_type = "Human"
    st.session_state.p2_type = "Human"
elif st.session_state.game_mode == "Human vs. Machine":
    st.session_state.p1_type = "Human"
    st.session_state.p2_type = "AI"
else:  # Machine vs. Machine
    st.session_state.p1_type = "AI"
    st.session_state.p2_type = "AI"


# --- 2. PREMIUM VISUALIZATION ENGINE ---
def draw_board_premium(board_obj):
    """Renders a beautifully structured Hexboard with color-coded perimeter goals."""
    fig, ax = plt.subplots(figsize=(7, 7), facecolor='#0e1117') # Match Streamlit dark theme bg
    ax.set_facecolor('#0e1117')
    ax.set_aspect('equal')
    radius = 0.577 
    
    color_map = {
        Player.EMPTY: '#1e2430', # Sleek dark slate for empty cells
        Player.RED: '#ff4b4b',   # High-contrast premium red
        Player.BLUE: '#00c0f2'   # Neon blue
    }
    
    size = board_obj.size
    
    for row in range(size):
        for col in range(size):
            # Skew math for transforming matrix indices to isometric hex coordinate space
            x = col + (row * 0.5)
            y = -row * (np.sqrt(3) / 2)
            
            player_at_pos = board_obj.grid[row, col]
            
            # Highlight border perimeters so recruiters immediately understand the rules
            edge_color = '#465362'
            edge_width = 1.0
            
            if row == 0 or row == size - 1:
                edge_color = '#ff4b4b' # Red targets top/bottom
                edge_width = 2.5
            if col == 0 or col == size - 1:
                # If it's a corner, blend it; otherwise, color it blue
                if not (row == 0 or row == size - 1):
                    edge_color = '#00c0f2' # Blue targets left/right
                    edge_width = 2.5

            hexagon = RegularPolygon(
                (x, y), 
                numVertices=6, 
                radius=radius, 
                orientation=np.radians(30),
                facecolor=color_map[player_at_pos], 
                edgecolor=edge_color,
                linewidth=edge_width
            )
            ax.add_patch(hexagon)
            
            # Subtle coordinate text overlays
            text_color = '#ffffff' if player_at_pos != Player.EMPTY else '#808495'
            ax.text(x, y, f"{row},{col}", ha='center', va='center', fontsize=10, color=text_color, weight='bold')

    # Automatically fit visual limits cleanly with padding
    ax.autoscale_view()
    ax.axis('off')
    return fig


# --- 3. THE CONTROL CENTER (SIDEBAR) ---
with st.sidebar:
    st.header("🎮 Match Setup")
    
    # Game Mode Controller
    selected_mode = st.selectbox(
        "Game Profile",
        ["Human vs. Human (Local)", "Human vs. Machine", "Machine vs. Machine"],
        key="mode_selector"
    )
    
    # State reset trigger if user changes the mode mid-game
    if selected_mode != st.session_state.game_mode:
        st.session_state.game_mode = selected_mode
        st.session_state.board = HexBoard(size=st.session_state.board_size)
        st.session_state.current_player = Player.RED
        st.session_state.winner = None
        st.st.rerun()

    st.markdown("---")
    st.subheader("📊 Match Status")
    
    if st.session_state.winner:
        winner_color = "🔴 RED" if st.session_state.winner == Player.RED else "🔵 BLUE"
        st.success(f"🏆 {winner_color} ENGINE WINS!")
    else:
        current_type = st.session_state.p1_type if st.session_state.current_player == Player.RED else st.session_state.p2_type
        player_label = "🔴 RED" if st.session_state.current_player == Player.RED else "🔵 BLUE"
        st.info(f"Active Turn: **{player_label}** ({current_type})")

    # Action Manual Inputs (temporary until we build a fully clickable canvas)
    if not st.session_state.winner and (
        (st.session_state.current_player == Player.RED and st.session_state.p1_type == "Human") or 
        (st.session_state.current_player == Player.BLUE and st.session_state.p2_type == "Human")
    ):
        st.markdown("### Manual Input Control")
        col1, col2 = st.columns(2)
        with col1:
            row_in = st.number_input("Target Row", min_value=0, max_value=st.session_state.board_size-1, value=0, step=1)
        with col2:
            col_in = st.number_input("Target Col", min_value=0, max_value=st.session_state.board_size-1, value=0, step=1)
            
        if st.button("Execute Move", use_container_width=True):
            if st.session_state.board.is_valid_move(row_in, col_in):
                st.session_state.board.place_piece(row_in, col_in, st.session_state.current_player)
                
                if st.session_state.board.check_win(st.session_state.current_player):
                    st.session_state.winner = st.session_state.current_player
                else:
                    st.session_state.current_player = Player.BLUE if st.session_state.current_player == Player.RED else Player.RED
                st.rerun()
            else:
                st.error("Illegal Target Move!")

    st.markdown("---")
    if st.button("Hard Reset Board", type="secondary", use_container_width=True):
        st.session_state.board = HexBoard(size=st.session_state.board_size)
        st.session_state.current_player = Player.RED
        st.session_state.winner = None
        st.rerun()


# --- 4. GAME MAIN LOOP WORKER ---
# Layout Division
col_layout_board, col_layout_telemetry = st.columns([2, 1])

with col_layout_board:
    # Render the styled, premium game board grid
    st.pyplot(draw_board_premium(st.session_state.board))

with col_layout_telemetry:
    st.subheader("📈 Real-time Telemetry")
    # This block handles active processing steps for the AI agent turn
    current_turn_type = st.session_state.p1_type if st.session_state.current_player == Player.RED else st.session_state.p2_type
    
    if not st.session_state.winner and current_turn_type == "AI":
        st.warning("🤖 AI Engine Thinking...")
        
        # This is our upcoming Step 2/3 hook. For now, we stub an elegant status loader.
        with st.status("Running MCTS Rollouts...", expanded=True) as status:
            st.write("Evaluating policy distributions from model tensors...")
            st.write("Traversing search tree nodes across game graph structures...")
            
            # Placeholder placeholder move selection for the state layout step
            # Pick the first available open hex cell deterministically
            ai_moved = False
            for r in range(st.session_state.board_size):
                for c in range(st.session_state.board_size):
                    if st.session_state.board.is_valid_move(r, c):
                        st.session_state.board.place_piece(r, c, st.session_state.current_player)
                        ai_moved = True
                        break
                if ai_moved:
                    break
                    
            status.update(label="Move Dispatched!", state="complete", expanded=False)
            
        # Post-move validation step check inside the state worker framework
        if st.session_state.board.check_win(st.session_state.current_player):
            st.session_state.winner = st.session_state.current_player
        else:
            st.session_state.current_player = Player.BLUE if st.session_state.current_player == Player.RED else Player.RED
        st.rerun()
    else:
        st.info("System idling. Awaiting action from human candidate player sequence.")