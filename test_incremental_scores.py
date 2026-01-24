import chess
import sys
from bot2 import Score
from timeit import default_timer

# ANSI color codes
RED = "\033[91m"
RESET = "\033[0m"


def print_score_details(score: Score, label=""):
    """Print detailed score information."""
    print(f"{label} Score Details:")
    print(f"  Material: {score.material}")
    print(f"  Midgame: {score.mg}")
    print(f"  Endgame: {score.eg}")
    print(f"  Non-pawn material: {score.npm}")
    print(f"  Pawn structure: {score.pawn_struct}")
    print(f"  King safety: {score.king_safety}")

    final_score = score.calculate()
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
    init_score = Score()
    init_score.initialize(board)
    print_score_details(init_score, "Initialized")

    # If a move is provided, test updating
    if move_uci:
        # Create a copy of the score for updating
        incremental_score = Score(init_score.material, init_score.mg, init_score.eg,
                                  init_score.npm, init_score.pawn_struct, init_score.king_safety)

        # Make the move and update the score
        move = chess.Move.from_uci(move_uci)
        incremental_score = incremental_score.updated(board, move)
        print_score_details(incremental_score, "After Incremental Update")

        # Initialize a fresh score for the resulting position
        board.push(move)
        fresh_score = Score()
        fresh_score.initialize(board)
        print_score_details(fresh_score, "Fresh Initialization of New Position")

        # Assert scores match
        assert (incremental_score.material ==
                fresh_score.material), f"Material scores do not match: {incremental_score.material} != {fresh_score.material}"
        assert (incremental_score.mg ==
                fresh_score.mg), f"Midgame scores do not match: {incremental_score.mg} != {fresh_score.mg}"
        assert (incremental_score.eg ==
                fresh_score.eg), f"Endgame scores do not match: {incremental_score.eg} != {fresh_score.eg}"
        assert (incremental_score.npm ==
                fresh_score.npm), f"Non-pawn material scores do not match: {incremental_score.npm} != {fresh_score.npm}"
        assert (incremental_score.pawn_struct ==
                fresh_score.pawn_struct), f"Pawn structure scores do not match: {incremental_score.pawn_struct} != {fresh_score.pawn_struct}"
        assert (incremental_score.king_safety ==
                fresh_score.king_safety), f"King safety scores do not match: {incremental_score.king_safety} != {fresh_score.king_safety}"

        incremental = incremental_score.calculate()
        fresh = fresh_score.calculate()
        assert (incremental == fresh), f"Final scores do not match: {incremental} != {fresh}"

    print("-" * 80)
    return board


if __name__ == "__main__":
    # Test individual positions from your game
    print("=== TESTING INDIVIDUAL POSITIONS ===")

    # Position 1: White to move
    fen1 = "r1b1r1k1/ppp2qbp/6p1/1N1p4/5B2/3p2P1/PPPK1p1P/R4Q1B w - - 0 25"
    board1 = test_position(fen1, "b5c7") # Knight captures pawn

    # Position 2: Black to move
    fen2 = "r1b1r1k1/ppN2qbp/6p1/3p4/5B2/3p2P1/PPPK1p1P/R4Q1B b - - 0 25"
    board2 = test_position(fen2, "c8h3") # Bishop moves to h3

    # Position 3: White to move
    fen3 = "r3r1k1/ppN2qbp/6p1/3p4/5B2/3p2Pb/PPPK1p1P/R4Q1B w - - 1 26"
    board3 = test_position(fen3, "f1h3") # Queen captures bishop

    # Position 4: Black to move
    # Black iso -> not iso
    # Black doubled -> not doubled
    fen4 = "r3r1k1/ppN2qbp/6p1/3p4/5B2/3p2PQ/PPPK1p1P/R6B b - - 0 26"
    board4 = test_position(fen4, "d3c2") # Pawn moves to c2

    # Position 5: White to move
    fen5 = "r3r1k1/ppN2qbp/6p1/3p4/5B2/6PQ/PPpK1p1P/R6B w - - 0 27"
    board5 = test_position(fen5, "c7a8") # Knight captures rook

    # Position 6
    fen6 = "r1bqkb1r/ppp1pppp/2n2n2/3p4/3P4/2N2N2/PPP1PPPP/R1BQKB1R w KQkq - 2 4"
    board6 = test_position(fen6, "c3d5") # Pawn captures pawn

    # Position 7: Test promoting a pawn
    fen7 = "8/1pp2p1P/8/1K6/8/5kP1/8/2r5 w - - 1 29"
    board7 = test_position(fen7, "h7h8q") # Pawn promotes to queen

    # Position 8: Self isolation (covid pawn)
    fen8 = "r1bq1rk1/pp3ppp/2p2n2/4n1N1/1b1p4/4p1N1/PPP1BPPP/R1BQ2KR w - - 0 12"
    board8 = test_position(fen8, "f2e3") # Pawn takes pawn

    # Position 9:
    fen9 = "r1b1k2r/pppp1p1p/4p1pB/4P3/3P3q/6P1/PPP2K1P/RN3BNR b kq - 0 12"
    board9 = test_position(fen9, "h4d4") # Queen takes pawn

    # Position 10:
    fen10 = "8/8/3kpNp1/4N3/4bKP1/6P1/2p5/3r4 b - - 1 52"
    board10 = test_position(fen10, "c2c1q") # Pawn promotes to queen

    # Position 11:
    fen11 = "r1bqkb1r/pppp1p1p/4p1pQ/4P3/3P4/8/PPP2nPP/RNB1KBNR w KQkq - 2 10"
    board11 = test_position(fen11, "e1f2")

    board = chess.Board(fen1)
    score = Score()
    score.initialize(board1)

    _ = score.calculate()

    import numpy as np

    start_time = default_timer()

    a = np.int16(-13212)
    b = np.int16(3492)

    n = 1_000_000
    for i in range(n):
        new_score = score.updated(board, chess.Move.from_uci("b5c7"))
        calculated = new_score.calculate()
        # calculated = score.numpy_calculate(board)
        d = np.maximum(a, b)
        c = np.int16(max(int(a), int(b)))
        if i % 100_000 == 0:
            print(f"Score after {i} moves: {calculated}")

    time_taken = default_timer() - start_time

    print(f"Time taken for {n} moves: {time_taken:.2f} seconds")
    print(f"Time per move: {time_taken / n * 1_000:.4f} ms")
