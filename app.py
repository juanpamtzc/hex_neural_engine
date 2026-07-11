import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import numpy as np

# Import your game engine
from src.engine.board import HexBoard, Player

# 1. Page Setup
st.set_page_config(page_title="Hex-Zero Arena", layout="centered")
st.title("⬡ Hex-Zero: Interactive Arena")
st.markdown("Red connects **Top-Left to Bottom-Right**. Blue connects **Top-Right to Bottom-Left**.")

# 2. Session State Initialization
if 'board' not in st.session_state:
    st.session_state.board = HexBoard(size=5)
    st.session_state.current_player = Player.RED
    st.session_state.winner = None

# 3. Matplotlib Rendering Function (Refactored for Physical Layout)
def draw_board(board_obj):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    radius = 0.5
    
    color_map = {
        Player.EMPTY: '#f0f2f6', # Light gray for empty to look like a physical board
        Player.RED: '#ff4b4b',
        Player.BLUE: '#1f77b4'
    }
    
    for row in range(board_obj.size):
        for col in range(board_obj.size):
            # Classic Diamond Projection (Flat-topped hexes)
            # Columns project down-right, Rows project down-left
            x = (col - row) * 1.5 * radius
            y = -(col + row) * (np.sqrt(3) / 2 * radius)
            
            player_at_pos = board_obj.grid[row, col]
            
            # orientation=0 forces flat-topped hexes, required for this diagonal interlocking
            hexagon = RegularPolygon(
                (x, y),
                numVertices=6,
                radius=radius * 0.95, # Slight gap between tiles for visual pop
                orientation=0, 
                facecolor=color_map[player_at_pos],
                edgecolor='black',
                linewidth=1.0
            )
            ax.add_patch(hexagon)
            
            # Coordinate labels
            text_color = 'white' if player_at_pos != Player.EMPTY else 'gray'
            ax.text(x, y, f"{row},{col}", ha='center', va='center', fontsize=8, color=text_color, fontweight='bold')

    ax.autoscale_view()
    ax.axis('off')
    return fig

# 4. Game UI & Logic
with st.sidebar:
    st.header("Make a Move")
    
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
st.pyplot(draw_board(st.session_state.board))