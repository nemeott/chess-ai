import chess
import sys
from bot2 import Score, MIDGAME, ENDGAME, PSQT, PIECE_VALUES_STOCKFISH, FLIP, NPM_SCALAR

# ANSI color codes
RED = "\033[91m"
RESET = "\033[0m"

def print_score_details(score, label=""):
    """Print detailed score information."""
    print(f"{label} Score Details:")
    print(f"  Material: {score.material}")
    print(f"  Midgame: {score.mg}")
    print(f"  Endgame: {score.eg}")
    print(f"  Non-pawn material: {score.npm}")
    print(f"  Pawn structure: {score.pawn_struct}")
    print(f"  King safety: {score.king_safety}")

    # Calculate final evaluation using the same formula as evaluate_position
    phase = min(score.npm // NPM_SCALAR, 256)
    interpolated_score = ((score.mg * phase) + (score.eg * (256 - phase))) >> 8
    final_score = score.material + interpolated_score
    print(f"  Final evaluation: {final_score}")
    print()

def highlight_diff(value):
    """Return a value with red highlighting if non-zero."""
    if value != 0:
        return f"{RED}{value}{RESET}"
    return value

def test_position(fen, move_uci=None):
    """Test a position with both initialize and update methods."""
    board = chess.Board(fen)

    print(f"Testing position: {fen}")
    if move_uci:
        print(f"Move to test: {move_uci}")
    print()

    # Initialize from scratch
    init_score = Score(0, 0, 0, 0, 0, 0)
    init_score.initialize_scores(board)
    print_score_details(init_score, "Initialized")

    # If a move is provided, test updating
    if move_uci:
        # Create a copy of the score for updating
        update_score = Score(init_score.material, init_score.mg, init_score.eg, init_score.npm, init_score.pawn_struct, init_score.king_safety)

        # Make the move and update the score
        move = chess.Move.from_uci(move_uci)
        update_score = update_score.updated(board, move)
        board.push(move)

        # Print the updated score
        print_score_details(update_score, "After Update")

        # Initialize a fresh score for the resulting position
        fresh_score = Score(0, 0, 0, 0, 0, 0)
        fresh_score.initialize_scores(board)
        print_score_details(fresh_score, "Fresh Initialization of New Position")

        # Compare the scores
        material_diff = update_score.material - fresh_score.material
        mg_diff = update_score.mg - fresh_score.mg
        eg_diff = update_score.eg - fresh_score.eg
        npm_diff = update_score.npm - fresh_score.npm
        pawn_struct_diff = update_score.pawn_struct - fresh_score.pawn_struct
        king_safety_diff = update_score.king_safety - fresh_score.king_safety

        print("Differences (Update - Fresh Init):")
        print(f"  Material: {highlight_diff(material_diff)}")
        print(f"  Midgame: {highlight_diff(mg_diff)}")
        print(f"  Endgame: {highlight_diff(eg_diff)}")
        print(f"  Non-pawn material: {highlight_diff(npm_diff)}")
        print(f"  Pawn structure: {highlight_diff(pawn_struct_diff)}")
        print(f"  King safety: {highlight_diff(king_safety_diff)}")

        print()

        assert(material_diff == 0)
        assert(mg_diff == 0)
        assert(eg_diff == 0)
        assert(npm_diff == 0)
        assert(pawn_struct_diff == 0)
        assert(king_safety_diff == 0)

    print("-" * 80)
    return board

if __name__ == "__main__":
    # Test individual positions from your game
    print("=== TESTING INDIVIDUAL POSITIONS ===")

    # Position 1: White to move
    fen1 = "r1b1r1k1/ppp2qbp/6p1/1N1p4/5B2/3p2P1/PPPK1p1P/R4Q1B w - - 0 25"
    board1 = test_position(fen1, "b5c7")  # Knight captures pawn

    # Position 2: Black to move
    fen2 = "r1b1r1k1/ppN2qbp/6p1/3p4/5B2/3p2P1/PPPK1p1P/R4Q1B b - - 0 25"
    board2 = test_position(fen2, "c8h3")  # Bishop moves to h3

    # Position 3: White to move
    fen3 = "r3r1k1/ppN2qbp/6p1/3p4/5B2/3p2Pb/PPPK1p1P/R4Q1B w - - 1 26"
    board3 = test_position(fen3, "f1h3")  # Queen captures bishop

    # Position 4: Black to move
    # Black iso -> not iso
    # Black doubled -> not doubled
    fen4 = "r3r1k1/ppN2qbp/6p1/3p4/5B2/3p2PQ/PPPK1p1P/R6B b - - 0 26"
    board4 = test_position(fen4, "d3c2")  # Pawn moves to c2

    # Position 5: White to move
    fen5 = "r3r1k1/ppN2qbp/6p1/3p4/5B2/6PQ/PPpK1p1P/R6B w - - 0 27"
    board5 = test_position(fen5, "c7a8")  # Knight captures rook

    # Position 6
    fen6 = "r1bqkb1r/ppp1pppp/2n2n2/3p4/3P4/2N2N2/PPP1PPPP/R1BQKB1R w KQkq - 2 4"
    board6 = test_position(fen6, "c3d5")  # Pawn captures pawn

    # Position 7: Test promoting a pawn
    fen7 = "8/1pp2p1P/8/1K6/8/5kP1/8/2r5 w - - 1 29"
    board7 = test_position(fen7, "h7h8q")  # Pawn promotes to queen

    # Position 8: Self isolation (covid pawn)
    fen8 = "r1bq1rk1/pp3ppp/2p2n2/4n1N1/1b1p4/4p1N1/PPP1BPPP/R1BQ2KR w - - 0 12"
    board8 = test_position(fen8, "f2e3")  # Pawn takes pawn
