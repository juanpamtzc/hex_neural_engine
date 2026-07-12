import pytest
from src.engine.board import HexBoard, Player

def test_red_win_vertical_straight():
    """Tests if Red wins by connecting Top to Bottom in a straight line."""
    board = HexBoard(size=3)
    
    board.place_piece(0, 1, Player.RED)
    board.place_piece(1, 1, Player.RED)
    assert not board.check_win(Player.RED), "Red should not win yet."
    
    board.place_piece(2, 1, Player.RED)
    assert board.check_win(Player.RED), "Red should win after connecting top to bottom."
    assert not board.check_win(Player.BLUE), "Blue should not trigger a win."

def test_blue_win_horizontal_straight():
    """Tests if Blue wins by connecting Left to Right in a straight line."""
    board = HexBoard(size=3)
    
    board.place_piece(1, 0, Player.BLUE)
    board.place_piece(1, 1, Player.BLUE)
    assert not board.check_win(Player.BLUE), "Blue should not win yet."
    
    board.place_piece(1, 2, Player.BLUE)
    assert board.check_win(Player.BLUE), "Blue should win after connecting left to right."
    assert not board.check_win(Player.RED), "Red should not trigger a win."

def test_winding_path_win():
    """Tests if the Union-Find correctly identifies a winding, non-linear win path."""
    board = HexBoard(size=4)
    
    # Red plays a zig-zag from Top to Bottom
    moves = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)]
    
    for i, (r, c) in enumerate(moves):
        board.place_piece(r, c, Player.RED)
        if i < len(moves) - 1:
            assert not board.check_win(Player.RED), f"Red falsely won at move {i}"
            
    assert board.check_win(Player.RED), "Red failed to win on a winding path."

def test_block_prevents_win():
    """Tests that an opponent's piece successfully blocks a Union-Find connection."""
    board = HexBoard(size=3)
    
    # Red tries to go down the middle
    board.place_piece(0, 1, Player.RED)
    board.place_piece(2, 1, Player.RED)
    
    # Blue blocks the middle
    board.place_piece(1, 1, Player.BLUE)
    
    assert not board.check_win(Player.RED), "Red should not win through a block."