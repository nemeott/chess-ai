import chess
from chess import polyglot # Polyglot for opening book
# from chess.polyglot import zobrist_hash # Built-in Zobrist hashing  TODO implement incremental hashing

from dataclasses import dataclass
import numpy as np

from typing_extensions import TypeAlias # For flags
from typing import Generator, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from game import ChessGame # Only import while type checking

from lru import LRU # For TT and history tables
from sys import getsizeof # For memory usage calculations

from constants import DEPTH, MAX_VALUE, MIN_VALUE, CHECKING_MOVE_ARROW, RENDER_DEPTH, TT_SIZE, PIECE_VALUES_STOCKFISH, \
    BISHOP_PAIR_BONUS, DOUBLED_PAWN_PENALTY, ISOLATED_PAWN_PENALTY, FLIP, MIDGAME, ENDGAME, PSQT, CASTLING_UPDATES, NPM_SCALAR, \
    OPENING_BOOK_PATH

import colors # Debug log colors
from timeit import default_timer # For debug timing

from numba import jit # (njit not needed since default is nopython since numba 0.59.0)


np.seterr(all="raise") # Raise warnings for all numpy errors

# Transposition table entry flags
Flag: TypeAlias = np.int8
EXACT: Flag = np.int8(1)
LOWERBOUND: Flag = np.int8(2) # Beta (fail-high)
UPPERBOUND: Flag = np.int8(3) # Alpha (fail-low)


@dataclass
class TTEntry:
    """
    Class to represent a transposition table entry.
    Stores the depth, value, flag, and best move for a position.
    """
    __slots__ = ["depth", "value", "flag", "best_move"] # Optimization for faster lookups

    depth: np.int8
    value: np.int16
    flag: Flag
    best_move: Optional[chess.Move]


@dataclass
class Score: # Positive values favor white, negative values favor black
    """
    Class to represent the score of a position.
    Stores material, midgame, endgame, non-pawn material, pawn structure, and king safety scores.
    Uses Numba for fast total score calculations.
    Has initialization and incremental update methods.
    Initialization is done once for the starting position and for checking if the incremental update is correct.
    Incremental updates are done for each move since it is much more efficient than re-evaluating the entire board (would have to push/pop each move).
    """
    # TODO: Move to own file
    __slots__ = ["material", "mg", "eg", "npm", "pawn_struct", "king_safety"] # Optimization for faster lookups

    def __init__(self, material: np.int16 = np.int16(0), mg: np.int16 = np.int16(0), eg: np.int16 = np.int16(0), npm: np.uint16 = np.uint16(0), pawn_struct: np.int8 = np.int8(0), king_safety: np.int8 = np.int8(0)) -> None:
        """
        Initialize the score with given values.
        """
        self.material: np.int16 = material # Material score
        self.mg: np.int16 = mg # Midgame score
        self.eg: np.int16 = eg # Endgame score
        self.npm: np.uint16 = npm # Non-pawn material (for phase calculation)
        self.pawn_struct: np.int8 = pawn_struct # Pawn structure score
        self.king_safety: np.int8 = king_safety # King safety score

    @staticmethod
    @jit(cache=True, fastmath=True)
    def _numba_calculate(material, mg, eg, npm, pawn_struct, king_safety) -> np.int16:
        """
        Calculate the total score for the position using Numba.
        Uses the phase to interpolate between midgame and endgame scores.
        Also uses the phase to weight the pawn structure score (more important in endgame).
        Adds the material score, interpolated mg/eg score, and interpolated pawn structure score.
        """
        # Phase value between 0 and 256 (0 = endgame, 256 = opening)
        phase = min(npm // NPM_SCALAR, 256)

        # Interpolate between midgame and endgame scores
        interpolated_mg_eg_score: int = ((mg * phase) + (eg * (256 - phase))) >> 8

        # Interpolate the pawn structure score (more important in endgame)
        interpolated_pawn_struct: int = (pawn_struct * (256 - phase)) >> 8

        return np.int16(material + interpolated_mg_eg_score + interpolated_pawn_struct)

    def calculate(self) -> np.int16:
        """
        Wrapper for the Numba calculate function.
        """
        return self._numba_calculate(self.material, self.mg, self.eg, self.npm, self.pawn_struct, self.king_safety)

    # def numpy_calculate(self, board: chess.Board) -> int:
    #     """
    #     UNUSED
    #     Attempt to use numpy for efficient calculations.
    #     (Definitely didn't work but left in case future optimizations possible)
    #     """
    #     # Cache tables for faster lookups
    #     mg_tables = PSQT[MIDGAME]
    #     eg_tables = PSQT[ENDGAME]

    #     material, mg, eg, npm, pawn_struct, king_safety = 0, 0, 0, 0, 0, 0
    #     for piece_type, piece_value in PIECE_VALUES_STOCKFISH.items():
    #         if piece_type != chess.KING:
    #             white_bitboard = board.pieces_mask(piece_type, chess.WHITE)
    #             black_bitboard = board.pieces_mask(piece_type, chess.BLACK)
    #             black_bitboard = chess.flip_horizontal(chess.flip_vertical(black_bitboard))

    #             white_bits = np.unpackbits(np.frombuffer(np.uint64(white_bitboard).tobytes(), dtype=np.uint8))
    #             black_bits = np.unpackbits(np.frombuffer(np.uint64(black_bitboard).tobytes(), dtype=np.uint8))

    #             # Material score
    #             white_material = (np.int16(white_bits) * piece_value).sum(dtype=np.int16)
    #             black_material = (np.int16(black_bits) * piece_value).sum(dtype=np.int16)
    #             material += white_material - black_material

    #             # Non-pawn material score
    #             if piece_type != chess.PAWN:
    #                 npm += white_material + black_material

    #             # Midgame and endgame scores
    #             mg += np.dot(white_bits, mg_tables[piece_type]) - np.dot(black_bits, mg_tables[piece_type]) # type: ignore
    #             eg += np.dot(white_bits, eg_tables[piece_type]) - np.dot(black_bits, eg_tables[piece_type]) # type: ignore

    #     # Phase value between 0 and 256 (0 = endgame, 256 = opening)
    #     phase = min(npm // NPM_SCALAR, 256) # type: ignore

    #     return material + (((mg * phase) + (eg * (256 - phase))) >> 8)

    def initialize(self, board: chess.Board) -> None:
        """
        Initialize values for a position (works with custom starting FENs).
        Calculates material score, midgame score, endgame score, npm score, pawn structure score, and king safety score.
        Evaluates piece positions using midgame and endgame piece-square tables with interpolation between them.
        Evaluates pawn structure for isolated and doubled pawns.
        Runs only once/minimally so not too optimized for clarity.
        """
        material, mg, eg, npm, pawn_struct, king_safety = 0, 0, 0, 0, 0, 0

        # Cache function for faster lookups
        _flip = FLIP

        # Cache tables for faster lookups
        _piece_values = PIECE_VALUES_STOCKFISH
        _mg_tables = PSQT[MIDGAME]
        _eg_tables = PSQT[ENDGAME]

        # Bishop counts for pair bonus
        white_bishop_count = 0
        black_bishop_count = 0

        # --- Material and Position Scores ---
        for square in chess.SQUARES:
            piece_type = board.piece_type_at(square)
            if piece_type:
                piece_color = board.color_at(square)

                # Update npm score
                if piece_type != chess.PAWN and piece_type != chess.KING:
                    npm += _piece_values[piece_type]

                # Update material and position scores
                if piece_color: # White piece
                    material += _piece_values[piece_type]
                    mg += _mg_tables[piece_type][_flip(square)] # type: ignore
                    eg += _eg_tables[piece_type][_flip(square)] # type: ignore
                    if piece_type == chess.BISHOP:
                        white_bishop_count += 1
                else: # Black piece
                    material -= _piece_values[piece_type]
                    mg -= _mg_tables[piece_type][square] # type: ignore
                    eg -= _eg_tables[piece_type][square] # type: ignore
                    if piece_type == chess.BISHOP:
                        black_bishop_count += 1

        # Bishop pair bonus worth half a pawn
        if white_bishop_count >= 2:
            material += BISHOP_PAIR_BONUS
        if black_bishop_count >= 2:
            material -= BISHOP_PAIR_BONUS

        # --- Pawn Structure ---
        file_masks = chess.BB_FILES

        # Process both colors at once with direct bitboard manipulation
        white_pawns = board.pieces_mask(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces_mask(chess.PAWN, chess.BLACK)

        # Evaluate isolated pawns for both colors in one pass
        for file in range(8): # TODO: Calculate pawn structure in previous loop
            # Pawns in this file
            white_pawns_in_file = chess.popcount(white_pawns & file_masks[file])
            black_pawns_in_file = chess.popcount(black_pawns & file_masks[file])

            # Calculate adjacent files mask once per iteration
            adjacent_mask = 0
            if file > 0:
                adjacent_mask |= file_masks[file - 1] # Append left file to mask
            if file < 7:
                adjacent_mask |= file_masks[file + 1] # Append right file to mask

            # Check for isolated pawns (no pawns in adjacent files)
            if white_pawns_in_file > 0 and chess.popcount(white_pawns & adjacent_mask) == 0:
                pawn_struct -= ISOLATED_PAWN_PENALTY # Isolated white pawn penalty
            if black_pawns_in_file > 0 and chess.popcount(black_pawns & adjacent_mask) == 0:
                pawn_struct += ISOLATED_PAWN_PENALTY # Isolated black pawn penalty

            # Check for doubled pawns
            if white_pawns_in_file > 1:
                pawn_struct -= DOUBLED_PAWN_PENALTY # Doubled white pawn penalty
            if black_pawns_in_file > 1:
                pawn_struct += DOUBLED_PAWN_PENALTY # Doubled black pawn penalty

        self.material, self.mg, self.eg, self.npm, self.pawn_struct, self.king_safety = \
            np.int16(material), np.int16(mg), np.int16(eg), np.uint16(npm), np.int8(pawn_struct), np.int8(king_safety)

    def updated(self, board: chess.Board, move: chess.Move) -> "Score": # TODO: Remove material since it is redundant with mg and eg
        """
        Returns the updated material, midgame, endgame, non-pawn material, pawn structure, and king safety scores based on the move.
        Much faster than re-evaluating the entire board, even if only the leaf nodes are re-evaluated.
        Hard to understand (and to write), but very worth it for performance.
        Incremental updates are much faster than re-evaluating the entire board since we would have to push and pop and iterate over the entire board.
        """
        material, mg, eg, npm, pawn_struct, king_safety = self.material, self.mg, self.eg, self.npm, self.pawn_struct, self.king_safety

        # Cache functions for faster lookups
        _popcount = chess.popcount
        _flip = FLIP

        # Cache tables for faster lookups
        _piece_values: dict[int, int] = PIECE_VALUES_STOCKFISH
        _mg_tables: list[Optional[np.ndarray]] = PSQT[MIDGAME]
        _eg_tables: list[Optional[np.ndarray]] = PSQT[ENDGAME]

        # Cache constants for faster lookups
        _isolated_pawn_penalty = ISOLATED_PAWN_PENALTY
        _doubled_pawn_penalty = DOUBLED_PAWN_PENALTY

        # Get move information
        from_square = move.from_square
        to_square = move.to_square
        promotion_piece_type: Optional[chess.PieceType] = move.promotion

        piece_type = board.piece_type_at(from_square) # ? Expensivish
        piece_color = board.turn
        color_multiplier = 1 if piece_color else -1 # 1 for white, -1 for black

        # --- Pawn Structure ---
        castling = False
        if piece_type == chess.PAWN: # Update pawn structure if moving a pawn
            pawns_before = board.pieces_mask(chess.PAWN, piece_color)
            pawns_after = pawns_before & ~(1 << from_square) # Remove moved pawn from pawns
            if promotion_piece_type: # If we are promoting, then we need to account for the disappearance of the pawn
                file_masks = chess.BB_FILES
                file = from_square & 7

                pawns_in_file_after = _popcount(pawns_after & file_masks[file])

                if pawns_in_file_after == 1: # 2 pawns in file before
                    pawn_struct += color_multiplier * _doubled_pawn_penalty # Remove doubled pawn penalty

                if pawns_in_file_after == 0: # No pawns in file after move
                    left_pawns = _popcount(pawns_after & file_masks[file - 1]) if file > 0 else 0
                    right_pawns = _popcount(pawns_after & file_masks[file + 1]) if file < 7 else 0

                    # If no longer isolated because of promotion
                    if left_pawns == 0 and right_pawns == 0:
                        pawn_struct += color_multiplier * _isolated_pawn_penalty # Remove penalty
                    # Check if left file is now isolated
                    if left_pawns >= 1 and (_popcount(pawns_after & file_masks[file - 2]) if file > 1 else 0) == 0:
                        pawn_struct -= color_multiplier * _isolated_pawn_penalty # Add penalty for isolated left file
                    # Check if right file is now isolated
                    if right_pawns >= 1 and (_popcount(pawns_after & file_masks[file + 2]) if file < 6 else 0) == 0:
                        pawn_struct -= color_multiplier * _isolated_pawn_penalty # Add penalty for isolated right file

            else: # Not promoting
                pawns_after |= 1 << to_square # Add moved pawn to pawns

                file_masks = chess.BB_FILES
                file = from_square & 7
                to_file = to_square & 7

                pawns_in_file_after = _popcount(pawns_after & file_masks[file])

                if file != to_file and pawns_in_file_after == 1: # 2 pawns in file before
                    pawn_struct += color_multiplier * _doubled_pawn_penalty # Remove doubled pawn penalty

                if to_file < file: # Move to left file (left, to_file, file, right)
                    left_pawns = _popcount(pawns_after & file_masks[to_file - 1]) if to_file > 0 else 0
                    pawns_in_to_file = _popcount(pawns_after & file_masks[to_file])
                    # pawns_in_file_after
                    right_pawns = _popcount(pawns_after & file_masks[file + 1]) if file < 7 else 0

                    if file != to_file and pawns_in_to_file == 2: # If now 2 pawns in file
                        pawn_struct -= color_multiplier * _doubled_pawn_penalty # Add doubled pawn penalty

                    # If no longer isolated because of move (moved left, had no right pawns)
                    if pawns_in_to_file == 1 and right_pawns == 0:
                        pawn_struct += color_multiplier * _isolated_pawn_penalty # Remove penalty

                    # Self isolating
                    if pawns_in_to_file >= 1 and left_pawns == 0 and pawns_in_file_after == 0:
                        pawn_struct -= color_multiplier * _isolated_pawn_penalty # Add penalty

                    # Left was isolated previously (have left pawns, added a pawn, and no pawns left of left adj)
                    if left_pawns >= 1 and pawns_in_to_file == 1 and (_popcount(pawns_after & file_masks[to_file - 2]) if to_file > 1 else 0) == 0:
                        pawn_struct += color_multiplier * _isolated_pawn_penalty # Remove penalty
                    # Right adj is now isolated
                    if pawns_in_file_after == 0 and right_pawns >= 1 and (_popcount(pawns_after & file_masks[file + 2]) if file < 6 else 0) == 0:
                        pawn_struct -= color_multiplier * _isolated_pawn_penalty # Add penalty

                elif to_file > file: # Move to right file (left, file, to_file, right)
                    left_pawns = _popcount(pawns_after & file_masks[file - 1]) if file > 0 else 0
                    # pawns_in_file_after
                    pawns_in_to_file = _popcount(pawns_after & file_masks[to_file])
                    right_pawns = _popcount(pawns_after & file_masks[to_file + 1]) if to_file < 7 else 0

                    if file != to_file and pawns_in_to_file == 2: # If now 2 pawns in file
                        pawn_struct -= color_multiplier * _doubled_pawn_penalty # Add doubled pawn penalty

                    # If no longer isolated because of move (moved right, had no left pawns)
                    if pawns_in_to_file == 1 and left_pawns == 0:
                        pawn_struct += color_multiplier * _isolated_pawn_penalty # Remove penalty

                    # Self isolating
                    if pawns_in_to_file >= 1 and right_pawns == 0 and pawns_in_file_after == 0:
                        pawn_struct -= color_multiplier * _isolated_pawn_penalty # Add penalty

                    # Left adj is now isolated
                    if pawns_in_file_after == 0 and left_pawns >= 1 and (_popcount(pawns_after & file_masks[file - 2]) if file > 1 else 0) == 0:
                        pawn_struct -= color_multiplier * _isolated_pawn_penalty # Add penalty
                    # Right was isolated previously (added a pawn, have right pawns, and no pawns right of right adj)
                    if right_pawns >= 1 and pawns_in_to_file == 1 and (_popcount(pawns_after & file_masks[to_file + 2]) if to_file < 6 else 0) == 0:
                        pawn_struct += color_multiplier * _isolated_pawn_penalty # Remove penalty

        # --- Castling ---
        elif piece_type == chess.KING: # Update rook scores if castling
            castle_info = CASTLING_UPDATES.get((from_square, to_square, piece_color))
            if castle_info:
                castling = True
                mg_rook_table = _mg_tables[chess.ROOK]
                eg_rook_table = _eg_tables[chess.ROOK]

                rook_from, rook_to = castle_info
                if piece_color: # Flip rook square for white
                    rook_from, rook_to = _flip(rook_from), _flip(rook_to)

                mg += color_multiplier * (mg_rook_table[rook_to] - mg_rook_table[rook_from]) # type: ignore
                eg += color_multiplier * (eg_rook_table[rook_to] - eg_rook_table[rook_from]) # type: ignore

        # --- Score Updates From Move ---
        new_from_square, new_to_square = from_square, to_square
        if piece_color: # Flip squares for white
            new_from_square, new_to_square = _flip(from_square), _flip(to_square)

        # Update position scores for moving piece
        if promotion_piece_type: # Promotion
            # Update bishop pair bonus if pawn promoted to bishop
            if promotion_piece_type == chess.BISHOP:
                bishop_count_before = board.pieces_mask(chess.BISHOP, piece_color).bit_count()
                if bishop_count_before == 1: # If 2 bishops now, add bonus
                    material += color_multiplier * BISHOP_PAIR_BONUS

            npm += _piece_values[promotion_piece_type]
            material += color_multiplier * (_piece_values[promotion_piece_type] - _piece_values[chess.PAWN])
            mg += color_multiplier * (_mg_tables[promotion_piece_type][new_to_square] - # type: ignore
                                      _mg_tables[chess.PAWN][new_from_square]) # type: ignore
            eg += color_multiplier * (_eg_tables[promotion_piece_type][new_to_square] - # type: ignore
                                      _eg_tables[chess.PAWN][new_from_square]) # type: ignore
        else: # Normal move
            mg_table = _mg_tables[piece_type] # type: ignore
            eg_table = _eg_tables[piece_type] # type: ignore
            mg += color_multiplier * (mg_table[new_to_square] - mg_table[new_from_square]) # ? Expensive
            eg += color_multiplier * (eg_table[new_to_square] - eg_table[new_from_square]) # ? Expensive (less)

        if castling: # Done if castling
            return Score(material, mg, eg, npm, pawn_struct, king_safety)

        # --- Capture ---
        captured_piece_type = board.piece_type_at(to_square) # ? Expensivish

        # Get en passant captured piece if applicable
        if not captured_piece_type and piece_type == chess.PAWN and board.is_en_passant(move):
            to_square -= color_multiplier * 8
            captured_piece_type = board.piece_type_at(from_square)

        if captured_piece_type: # Piece captured
            # --- Pawn Structure ---
            if captured_piece_type == chess.PAWN: # Capturing a pawn
                enemy_pawns_before = board.pieces_mask(chess.PAWN, not piece_color)
                enemy_pawns_after = enemy_pawns_before & ~(1 << to_square) # Remove captured pawn from enemy pawns

                file_masks = chess.BB_FILES
                file = to_square & 7 # Get the file of the captured pawn

                pawns_in_file_after = _popcount(enemy_pawns_after & file_masks[file])
                if pawns_in_file_after == 0: # No pawns in file after move
                    left_pawns = _popcount(enemy_pawns_after & file_masks[file - 1]) if file > 0 else 0
                    right_pawns = _popcount(enemy_pawns_after & file_masks[file + 1]) if file < 7 else 0

                    # Update isolated pawn penalties
                    if left_pawns == 0 and right_pawns == 0: # Pawn isolated previously
                        pawn_struct += -color_multiplier * _isolated_pawn_penalty # Remove penalty
                    # Left adj isolated
                    if left_pawns >= 1 and (_popcount(enemy_pawns_after & file_masks[file - 2]) if file > 1 else 0) == 0:
                        pawn_struct -= -color_multiplier * _isolated_pawn_penalty # Add penalty
                    # Right adj isolated
                    if right_pawns >= 1 and (_popcount(enemy_pawns_after & file_masks[file + 2]) if file < 6 else 0) == 0:
                        pawn_struct -= -color_multiplier * 20 # Add penalty
                elif pawns_in_file_after == 1: # 2 pawns in file before
                    pawn_struct += -color_multiplier * _doubled_pawn_penalty # Remove doubled pawn penalty

            # --- Score Updates From Capture ---
            else: # Capturing a piece other than a pawn
                # Update npm score
                npm -= _piece_values[captured_piece_type]

                # Update bishop pair bonus if bishop captured
                if captured_piece_type == chess.BISHOP and board.pieces_mask(captured_piece_type, not piece_color).bit_count() == 2:
                    material -= -color_multiplier * BISHOP_PAIR_BONUS # If 2 bishops before, remove bonus

            if not piece_color: # Flip to square if black capturing white
                to_square = _flip(to_square)

            # Remove captured piece from material and position scores
            material -= -color_multiplier * _piece_values[captured_piece_type]
            mg -= -color_multiplier * _mg_tables[captured_piece_type][to_square] # type: ignore
            eg -= -color_multiplier * _eg_tables[captured_piece_type][to_square] # type: ignore

        return Score(material, mg, eg, npm, pawn_struct, king_safety) # ? Expensive


class ChessBot:
    """
    Class to represent the chess bot.
    """
    __slots__ = ["game", "moves_checked", "transposition_table", "opening_book"] # Optimization for fast lookups

    def __init__(self, game, use_opening_book: bool = True) -> None:
        """
        Initialize the chess bot with the game instance.
        Also initializes the transposition table with size in MB.
        """
        self.game: "ChessGame" = game
        self.moves_checked: int = 0

        # Initialize transposition table with size in MB
        tt_entry_size = getsizeof(TTEntry(np.int8(0), np.int16(0), EXACT, chess.Move.from_uci("e2e4")), 64)
        self.transposition_table = LRU(int(TT_SIZE) * 1024 * 1024 // tt_entry_size) # Initialize TT with size in MB

        # Initialize opening book
        self.opening_book = None
        if use_opening_book and OPENING_BOOK_PATH:
            try:
                self.opening_book = polyglot.open_reader(OPENING_BOOK_PATH)
                print(f"{colors.GREEN}Opening book loaded successfully.{colors.RESET}")
            except Exception as e:
                print(f"{colors.RED}Error loading opening book: {e}{colors.RESET}")

    def display_checking_move_arrow(self, move) -> None:
        """
        Display an arrow on the board for the move being checked.
        Used for debugging purposes.
        """
        self.game.checking_move = move
        self.game.display_board(self.game.last_move) # Update display

    def increment_moves_and_render_arrow(self, depth: np.int8, move: chess.Move) -> None:
        self.moves_checked += 1
        if CHECKING_MOVE_ARROW and depth >= RENDER_DEPTH: # Display the root move
            self.display_checking_move_arrow(move)

    def evaluate_position(self, board: chess.Board, score: Score, tt_entry: Optional[TTEntry] = None, has_legal_moves: bool = True) -> np.int16:
        """
        Evaluate the current position.
        Positive values favor white, negative values favor black.
        """
        if tt_entry:
            return tt_entry.value

        # Check expensive operations once
        if has_legal_moves:
            has_legal_moves = any(board.generate_legal_moves()) # ! SLOW

        # Evaluate game-ending conditions
        if not has_legal_moves: # No legal moves
            if board.is_check(): # Checkmate
                return MIN_VALUE if board.turn else MAX_VALUE
            return np.int16(0) # Stalemate
        elif board.is_insufficient_material():
            return np.int16(0)
        elif board.can_claim_fifty_moves(): # Avoid fifty move rule
            return np.int16(0)

        return score.calculate()

    def ordered_moves_generator(self, board: chess.Board, tt_move: Optional[chess.Move]) -> Generator[chess.Move, None, None]:
        """
        Generate ordered moves for the current position.
        Uses a simple heuristic to order moves based on piece values and captures.
        """
        # Yield transposition table move first
        if tt_move:
            yield tt_move

        # Cache functions for faster lookups
        _is_capture = board.is_capture
        _piece_type_at = board.piece_type_at

        # Cache table for faster lookups
        _piece_values = PIECE_VALUES_STOCKFISH

        color_multiplier = 1 if board.turn else -1 # 1 for white, -1 for black

        # Sort remaining moves
        ordered_moves = []
        for move in board.generate_legal_moves(): # ! REALLY SLOW
            if not tt_move or move != tt_move: # Skip TT move since already yielded
                score = 0

                # Capturing a piece bonus (MVV/LVA - Most Valuable Victim/Least Valuable Attacker)
                if _is_capture(move):
                    victim_piece_type = _piece_type_at(move.to_square)
                    attacker_piece_type = _piece_type_at(move.from_square)

                    # Handle en passant captures
                    if not victim_piece_type and attacker_piece_type == chess.PAWN: # Implied en passant capture since no piece at to_square and pawn moving
                        victim_piece_type = _piece_type_at(move.to_square - (color_multiplier * 8))
                        score += 5 # Small bonus for en passant captures

                    # Prioritize capturing higher value pieces using lower value pieces
                    score += 10_000 + (_piece_values[victim_piece_type] - # type: ignore
                                       _piece_values[attacker_piece_type]) # type: ignore

                if move.promotion: # Promotion bonus
                    score += 1_000 + _piece_values[move.promotion] - _piece_values[chess.PAWN]

                # if board.gives_check(move): # ! SLOW
                #     score += 100

                ordered_moves.append((move, score))

        ordered_moves.sort(key=lambda x: x[1], reverse=True)
        # print(len(ordered_moves)) if len(ordered_moves) > 30 else None

        for move_and_score in ordered_moves:
            yield move_and_score[0]

    def quiescence(self, board: chess.Board, depth, alpha, beta, score) -> np.int16:
        """
        Quiescence search to avoid horizon effect.
        Searches only captures until a quiet position is reached.
        """
        # self.moves_checked += 1

        # Cache functions for faster lookups
        _piece_type_at = board.piece_type_at
        _push = board.push
        _pop = board.pop

        # Cache table for faster lookups
        _piece_values = PIECE_VALUES_STOCKFISH

        color_multiplier = 1 if board.turn else -1 # 1 for white, -1 for black

        # Evaluate position (lazy evaluation)
        stand_pat = score.calculate()

        if stand_pat >= beta: # Beta cutoff
            return beta # TODO: Test vs returning stand pat
        if alpha < stand_pat: # Update alpha if stand pat is better
            alpha = stand_pat

        for move in board.generate_legal_captures():
            # Skip if capture is not worth it
            if not move.promotion:
                victim_piece_type = _piece_type_at(move.to_square)
                if not victim_piece_type and board.is_en_passant(move):
                    victim_piece_type = _piece_type_at(move.to_square - (color_multiplier * 8))

                if stand_pat + _piece_values[victim_piece_type] <= alpha: # type: ignore
                    continue

            updated_score: Score = score.updated(board, move)

            _push(move)
            value = -self.quiescence(board, depth - 1, -beta, -alpha, updated_score)
            _pop()

            if value >= beta: # Beta cutoff
                return beta
            if value > alpha: # Update alpha if value is better
                alpha = value

        return alpha # Return the best value found

    def alpha_beta(self, board: chess.Board, depth: np.int8, alpha: float, beta: float, maximizing_player: bool, score: Score, allow_null_move: bool = True) -> tuple[np.int16, Optional[chess.Move]]:
        """
        Fail-soft alpha-beta search with transposition table.
        Scores are incrementally updated based on the move.
        Returns the best value and move for the current player.
        TODO, PV search, iterative deepening, quiescence search, killer moves, history heuristic, late move reduction, null move pruning.
        """
        # Cache functions for faster lookups
        _push = board.push
        _pop = board.pop
        _score_updated = score.updated
        _increment_moves_and_render_arrow = self.increment_moves_and_render_arrow

        original_alpha, original_beta = alpha, beta

        # Lookup position in transposition table
        # key = zobrist_hash(board) # ! REALLY SLOW (because it is not incremental)
        key = board._transposition_key() # ? Much faster
        tt_entry: Optional[TTEntry] = self.transposition_table.get(key) # TODO: Check if actually getting best move

        # If position is in transposition table and depth is sufficient
        tt_move = None
        if tt_entry and tt_entry.depth >= depth: # TODO: Check vs depth < depth??
            tt_move = tt_entry.best_move
            if tt_entry.flag == EXACT:
                return tt_entry.value, tt_move
            elif tt_entry.flag == LOWERBOUND and tt_entry.value >= beta:
                return tt_entry.value, tt_move
            elif tt_entry.flag == UPPERBOUND and tt_entry.value <= alpha:
                return tt_entry.value, tt_move

        # Terminal node check
        if depth == 0:
            value = self.evaluate_position(board, score, tt_entry)
            return value, None # No move to return
            # return self.quiescence(board, 3, alpha, beta, score), None # No move to return

        # --- Null Move Pruning ---
        # if allow_null_move and depth >= 3 and not board.is_check(): # Depth sufficient, and not in check
        #     bitboard: chess.Bitboard = board.occupied_co[board.turn]
        #     bitboard = bitboard & ~board.pawns & ~board.kings # Remove pawns and kings from bitboard
        #     if bitboard > 0: # If there are non-pawn pieces
        #         R = 3 if score.npm < 1_500 else 2 # Reduction factor

        #         _push(chess.Move.null()) # Make null move
        #         null_value: np.int16 = -self.alpha_beta(board, np.int8(depth - R - 1), -beta, -beta + 1, not maximizing_player, score, False)[0]
        #         _pop() # Undo null move

        #         # If null move causes beta cutoff, prune this subtree
        #         if null_value >= beta:
        #             return null_value, None

        best_move = None
        if maximizing_player:
            best_value: np.int16 = MIN_VALUE
            for move in self.ordered_moves_generator(board, tt_move):
                _increment_moves_and_render_arrow(depth, move) # Increment moves checked and render arrow

                updated_score: Score = _score_updated(board, move)
                _push(move)
                value: np.int16 = self.alpha_beta(board, depth - 1, alpha, beta, False, updated_score)[0]
                _pop()

                if value > best_value: # Get new best value and move
                    best_value, best_move = value, move
                    alpha = max(int(alpha), int(best_value)) # Get new alpha
                    if alpha >= beta:
                        break # Beta cutoff (fail-high: opponent won't allow this position)

        else: # Minimizing player
            best_value: np.int16 = MAX_VALUE
            for move in self.ordered_moves_generator(board, tt_move):
                _increment_moves_and_render_arrow(depth, move) # Increment moves checked and render arrow

                updated_score: Score = _score_updated(board, move)
                _push(move)
                value: np.int16 = self.alpha_beta(board, depth - 1, alpha, beta, True, updated_score)[0]
                _pop()

                if value < best_value: # Get new best value and move
                    best_value, best_move = value, move
                    beta = min(int(beta), int(best_value)) # Get new beta
                    if beta <= alpha:
                        break # Alpha cutoff (fail-low: other positions are better)

        if best_move is None: # If no legal moves, evaluate position
            return self.evaluate_position(board, score, tt_entry, has_legal_moves=False), None

        # Store position in transposition table
        if best_value <= original_alpha:
            flag = UPPERBOUND
        elif best_value >= original_beta:
            flag = LOWERBOUND
        else:
            flag = EXACT

        self.transposition_table[key] = TTEntry(depth, best_value, flag, best_move)

        return best_value, best_move

    # TODO WIP ---------------------------------------------

    def next_guess(self, alpha, beta, subtree_count):
        return alpha + (beta - alpha) * (subtree_count - 1) / subtree_count

    def best_node_search(self, board: chess.Board, alpha, beta, maximizing_player: bool):
        """
        Experimental best node search (fuzzified game search) algorithm based on the paper by Dmitrijs Rutko.
        Uses the next guess function to return the separation value for the next iteration.
        """
        alpha, beta = float(alpha), float(beta)

        ordered_moves = list(self.ordered_moves_generator(board, None))
        subtree_count = len(ordered_moves)
        color_multiplier = 1 if maximizing_player else -1

        original_score = self.game.score
        better_count = 0

        best_move = None
        best_value = None
        while beta - alpha >= 2 and better_count != 1:
            separation_value = self.next_guess(alpha, beta, subtree_count)

            better_count = 0
            better = []
            for move in ordered_moves:
                self.increment_moves_and_render_arrow(DEPTH, move)

                score = original_score.updated(board, move)
                board.push(move)
                move_value = self.alpha_beta(board,
                                             DEPTH - 1,
                                             -(separation_value), # TODO -(sep_value+1) and other variations
                                             -(separation_value - 1),
                                             not maximizing_player,
                                             score)[0]
                board.pop()

                if color_multiplier * int(move_value) >= separation_value:
                    better_count += 1
                    better.append(move)
                    best_move = move
                    best_value = move_value

                    # expected_quality = 1 / ((better_count / self.moves_checked) * (subtree_count**int(DEPTH)))
                    # if expected_quality > 0.95:
                    #     return best_value, best_move

            # Update alpha-beta range
            if better_count > 1:
                alpha = separation_value

                # Update number of sub-trees that exceeds separation test value
                if subtree_count != better_count:
                    subtree_count = better_count
                    ordered_moves = better
            else:
                beta = separation_value

        return best_value, best_move
    
    def iterative_deepening_mdt_f(self, board: chess.Board) -> tuple[np.int16, Optional[chess.Move]]:
        first_guess, best_move = np.int16(0), None
        for depth in range(1, DEPTH + 1):
            first_guess, best_move = self.mtd_f(board, first_guess)

        return first_guess, best_move

    def mtd_f(self, board: chess.Board, f) -> tuple[np.int16, Optional[chess.Move]]:
        g = f
        upper_bound = MAX_VALUE
        lower_bound = MIN_VALUE

        best_move = None
        while lower_bound < upper_bound:
            beta = max(int(g), int(lower_bound) + 1)
            
            g, best_move = self.alpha_beta(board, DEPTH, beta - 1, beta, board.turn, self.game.score)
            if g < beta:
                upper_bound = g 
            else:
                lower_bound = g

        return g, best_move

    def get_move(self, board: chess.Board):
        """
        Main method to get the best move for the current player.
        """
        self.moves_checked = 0

        alpha, beta = float(MIN_VALUE), float(MAX_VALUE)

        time_taken: float = 0.0
        best_move: Optional[chess.Move] = chess.Move.null()
        if self.opening_book:
            book_entry = self.opening_book.get(board) # Get the best book move
            if book_entry: # Use opening book move
                best_move = book_entry.move

        if not best_move: # No book move found, use alpha-beta search
            start_time: float = default_timer() # Start timer

            # best_value, best_move = self.alpha_beta(board, DEPTH, alpha, beta, board.turn, self.game.score)
            # best_value, best_move = self.best_node_search(board, alpha, beta, board.turn)
            best_value, best_move = self.iterative_deepening_mdt_f(board)

            time_taken = default_timer() - start_time # Stop timer

            print(f"Goal value: {best_value}")

        if best_move is None:
            legal_moves = list(board.generate_legal_moves())
            if len(legal_moves) > 0:
                best_move = legal_moves[0]
            else:
                print(f"{colors.RED}No best move returned{colors.RESET}")
                print(f"{colors.RED}Legal moves: {legal_moves}{colors.RESET}")
                quit()

        self.game.score = self.game.score.updated(board, best_move) # type: ignore

        self.print_stats(board, time_taken)

        return best_move
    
    def print_stats(self, board: chess.Board, time_taken: float) -> None:
        """
        Print statistics about the search.
        Prints the number of moves checked, time taken, moves per second, and transposition table size.
        """
        # Moves checked over time taken
        time_per_move = time_taken / self.moves_checked if self.moves_checked > 0 else 0
        moves_per_second = 1 / time_per_move if time_per_move > 0 else 0
        print(f"Moves/Time: {colors.BOLD}{colors.get_moves_color(self.moves_checked)}{self.moves_checked:,}{colors.RESET} / "
              f"{colors.BOLD}{colors.get_move_time_color(time_taken)}{time_taken:.2f}{colors.RESET} s = "
              f"{colors.BOLD}{colors.CYAN}{time_per_move * 1000:.4f}{colors.RESET} ms/M, "
              f"{colors.BOLD}{colors.CYAN}{moves_per_second:,.0f}{colors.RESET} M/s")

        # Calculate memory usage more accurately
        tt_entry_size = getsizeof(TTEntry(np.int8(0), np.int16(0), EXACT, chess.Move.from_uci("e2e4")), 64)
        transposition_table_entries = len(self.transposition_table)
        tt_size_mb = transposition_table_entries * tt_entry_size / (1024 * 1024)
        # eval_size_mb = sum(getsizeof(k) + getsizeof(v) for k, v in list(self.evaluation_cache.items())[:10]) / 10
        # eval_size_mb = eval_size_mb * len(self.evaluation_cache) / (1024 * 1024)

        # Print cache statistics
        print(f"Transposition table: {colors.BOLD}{colors.MAGENTA}{transposition_table_entries:,}{colors.RESET} entries, "
              f"{colors.BOLD}{colors.CYAN}{tt_size_mb:.4f}{colors.RESET} MB")
        # print(f"Evaluation cache: {colors.BOLD}{colors.MAGENTA}{len(self.evaluation_cache):,}{colors.RESET} entries, "
        #       f"{colors.BOLD}{colors.CYAN}{eval_size_mb:.4f}{colors.RESET} MB")

        # Print the FEN
        print(f"FEN: {board.fen()}")
