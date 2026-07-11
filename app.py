import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import numpy as np
import random

# Import your core engine structures
from src.engine.board import HexBoard, Player

# --- 1. PAGE SETUP & THEME ---
st.set_page_config(
    page_title="Hex-Zero Arena",
    page_icon="⬢",
    layout="wide"
)

st.title("⬢ Hex-Zero: Reinforcement Learning Arena")
st.markdown("### Production Front-End Framework")

# --- 2. CONFIGURABLE STATE MANAGEMENT ---
# Initialize persistent game state variables
if 'board_size' not in st.session_state:
    st.session_state.board_size = 5  # Matches your default board.py size

if 'board' not in st.session_state:
    st.session_state.board = HexBoard(size=st.session_state.board_size)

if 'current_player' not in st.session_state:
    st.session_state.current_player = Player.RED

if 'winner' not in st.session_state:
    st.session_state.winner = None

if 'game_mode' not in st.session_state:
    st.session_state.game_mode = "Human vs. Machine"

# Map player structural identities based on the selected game mode
if st.session_state.game_mode == "Human vs. Human (Local)":
    st.session_state.p1_type = "Human"
    st.session_state.p2_type = "Human"
elif st.session_state.game_mode == "Human vs. Machine":
    st.session_state.p1_type = "Human"
    st.session_state.p2_type = "AI"
else:  # Machine vs. Machine
    st.session_state.p1_type = "AI"
    st.session_state.p2_type = "AI"


# --- 3. PREMIUM VISUALIZATION ENGINE ---
def draw_board_premium(board_obj):
    """Renders a clean, high-contrast Hex board highlighting the perimeter targets."""
    # Use a clean dark background format for maximum contrast
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='#0e1117')
    ax.set_facecolor('#0e1117')
    ax.set_aspect('equal')
    radius = 0.577 
    
    color_map = {
        Player.EMPTY: '#1e2430',  # Slate dark for empty cells
        Player.RED: '#ff4b4b',    # Clean Red
        Player.BLUE: '#1f77b4'    # Clean Blue
    }
    
    size = board_obj.size
    
    for row in range(size):
        for col in range(size):
            # Skew math for transforming matrix indices to isometric hex coordinate space
            x = col + (row * 0.5)
            y = -row * (np.sqrt(3) / 2)  # Negative keeps row 0 at the top physically
            
            player_at_pos = board_obj.grid[row, col]
            
            # Highlight border perimeters so win trajectories make visual sense
            edge_color = '#465362'
            edge_width = 1.2
            
            # Red targets Top and Bottom edges
            if row == 0 or row == size - 1:
                edge_color = '#ff4b4b'
                edge_width = 2.5
            # Blue targets Left and Right edges
            if col == 0 or col == size - 1:
                if not (row == 0 or row == size - 1): # Don't overwrite corners entirely
                    edge_color = '#1f77b4'
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
            
            # Dynamic coordinate text color based on cell occupancy
            text_color = '#ffffff' if player_at_pos != Player.EMPTY else '#808495'
            ax.text(x, y, f"{row},{col}", ha='center', va='center', fontsize=9, color=text_color, weight='bold')

    ax.autoscale_view()
    ax.axis('off')
    return fig


# --- 4. SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.header("🎮 Match Setup")
    
    # Game Profile Select Box
    selected_mode = st.selectbox(
        "Game Mode",
        ["Human vs. Human (Local)", "Human vs. Machine", "Machine vs. Machine"],
        index=["Human vs. Human (Local)", "Human vs. Machine", "Machine vs. Machine"].index(st.session_state.game_mode)
    )
    
    # Reset match gracefully if game mode is changed mid-session
    if selected_mode != st.session_state.game_mode:
        st.session_state.game_mode = selected_mode
        st.session_state.board = HexBoard(size=st.session_state.board_size)
        st.session_state.current_player = Player.RED
        st.session_state.winner = None
        st.rerun()

    st.markdown("---")
    st.subheader("📊 Live Status")
    
    if st.session_state.winner:
        winner_color = "🔴 RED" if st.session_state.winner == Player.RED else "🔵 BLUE"
        st.success(f"🏆 {winner_color} WINS!")
    else:
        active_type = st.session_state.p1_type if st.session_state.current_player == Player.RED else st.session_state.p2_type
        player_label = "🔴 RED" if st.session_state.current_player == Player.RED else "🔵 BLUE"
        st.info(f"Turn: **{player_label}** ({active_type})")

    # Manual Coordinate Inputs for Human Players
    current_turn_type = st.session_state.p1_type if st.session_state.current_player == Player.RED else st.session_state.p2_type
    if not st.session_state.winner and current_turn_type == "Human":
        st.markdown("### Manual Play Input")
        col1, col2 = st.columns(2)
        with col1:
            row_in = st.number_input("Row", min_value=0, max_value=st.session_state.board_size-1, value=0, step=1)
        with col2:
            col_in = st.number_input("Col", min_value=0, max_value=st.session_state.board_size-1, value=0, step=1)
            
        if st.button("Place Piece", use_container_width=True, type="primary"):
            if st.session_state.board.is_valid_move(row_in, col_in):
                st.session_state.board.place_piece(row_in, col_in, st.session_state.current_player)
                
                if st.session_state.board.check_win(st.session_state.current_player):
                    st.session_state.winner = st.session_state.current_player
                else:
                    st.session_state.current_player = Player.BLUE if st.session_state.current_player == Player.RED else Player.RED
                st.rerun()
            else:
                st.error("Invalid move! Cell already occupied.")

    st.markdown("---")
    if st.button("Reset Game board", use_container_width=True):
        st.session_state.board = HexBoard(size=st.session_state.board_size)
        st.session_state.current_player = Player.RED
        st.session_state.winner = None
        st.rerun()


# --- 5. MAIN GRAPHICS AND AUTOMATED GAME LOOP ---
layout_board, layout_telemetry = st.columns([2, 1])

with layout_board:
    # Display our sleek premium board
    st.pyplot(draw_board_premium(st.session_state.board))

with layout_telemetry:
    st.subheader("🤖 Engine Output Log")
    
    # If it is an AI's turn, compute the automated next move
    if not st.session_state.winner and current_turn_type == "AI":
        with st.status("Engine computing optimal play matrix...", expanded=True) as status:
            st.write("Extracting valid action coordinates...")
            
            # Find all open cells
            valid_moves = []
            for r in range(st.session_state.board_size):
                for c in range(st.session_state.board_size):
                    if st.session_state.board.is_valid_move(r, c):
                        valid_moves.append((r, c))
            
            # Select a random move for now as our agent stub
            if valid_moves:
                ai_row, ai_col = random.choice(valid_moves)
                st.write(f"Executing placeholder evaluation play at: ({ai_row}, {ai_col})")
                st.session_state.board.place_piece(ai_row, ai_col, st.session_state.current_player)
            
            status.update(label="Move committed!", state="complete")
            
        # Post-move evaluation check
        if st.session_state.board.check_win(st.session_state.current_player):
            st.session_state.winner = st.session_state.current_player
        else:
            st.session_state.current_player = Player.BLUE if st.session_state.current_player == Player.RED else Player.RED
        
        # Short pause for Machine vs. Machine simulation observation, then rerun
        import time
        if st.session_state.game_mode == "Machine vs. Machine":
            time.sleep(0.4)
        st.rerun()
    else:
        if st.session_state.winner:
            st.success("Match complete.")
        else:
            st.info("System idling. Awaiting action from human player.")