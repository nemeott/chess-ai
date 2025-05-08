import chess
# from board import ChessBoard
from chess.polyglot import zobrist_hash # Built-in Zobrist hashing  TODO implement incremental hashing

from dataclasses import dataclass  # For TT entries and scores
from typing_extensions import TypeAlias  # For flags
import numpy as np
from typing import Generator, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from game import ChessGame  # Only import while type checking

from lru import LRU  # For TT and history tables
from sys import getsizeof # For memory usage

from constants import DEPTH, MAX_VALUE, MIN_VALUE, CHECKING_MOVE_ARROW, RENDER_DEPTH, TT_SIZE, \
    PIECE_VALUES_STOCKFISH, BISHOP_PAIR_BONUS, FLIP, MIDGAME, ENDGAME, PSQT, CASTLING_UPDATES, NPM_SCALAR

import colors  # Debug log colors
from timeit import default_timer # For debugging timing

# from numba import njit, vectorize


# Transposition table entry flags
Flag: TypeAlias = np.int8
EXACT: Flag = np.int8(1)
LOWERBOUND: Flag = np.int8(2)  # Beta (fail-high)
UPPERBOUND: Flag = np.int8(3)  # Alpha (fail-low)

# Transposition table entry
@dataclass
class TTEntry:
    __slots__ = ["depth", "value", "flag", "best_move"] # Optimization for faster lookups

    depth: np.int8
    value: np.int16
    flag: Flag
    best_move: Optional[chess.Move]

# TODO: Move to own file
# Score class to initialize and update scores
@dataclass
class Score: # Positive values favor white, negative values favor black
    __slots__ = ["material", "mg", "eg", "npm", "pawn_struct", "king_safety"] # Optimization for faster lookups

    material: np.int16 # Material score
    mg: np.int16 # Midgame score
    eg: np.int16 # Endgame score
    npm: np.uint16 # Non-pawn material (for phase calculation)
    pawn_struct: np.int8 # Pawn structure score TODO: Check for overflow
    king_safety: np.int8 # King safety score

    def __init__(self, material: np.int16 = np.int16(0), mg: np.int16 = np.int16(0), eg: np.int16 = np.int16(0), npm: np.uint16 = np.uint16(0), pawn_struct: np.int8 = np.int8(0), king_safety: np.int8 = np.int8(0)) -> None:
        """
        Initialize the score with given values.
        """
        self.material = material
        self.mg = mg
        self.eg = eg
        self.npm = npm
        self.pawn_struct = pawn_struct
        self.king_safety = king_safety

    def calculate(self) -> np.int16:
        """
        Calculate the score for the current position.
        Uses the phase to interpolate between midgame and endgame scores.
        Also uses the phase to weight the pawn structure score (more important in endgame).
        Adds the material score, interpolated mg/eg score, and interpolated pawn structure score.
        """
        # Phase value between 0 and 256 (0 = endgame, 256 = opening)
        phase = min(self.npm // NPM_SCALAR, 256)
        # assert 0 <= phase <= 256, f"Phase value out of bounds: {phase}"

        # Interpolate between midgame and endgame scores
        interpolated_mg_eg_score: int = ((int(self.mg) * phase) + (int(self.eg) * (256 - phase))) >> 8 

        # Interpolate the pawn structure score (more important in endgame)
        interpolated_pawn_struct: int = (int(self.pawn_struct) * (256 - phase)) >> 8

        return self.material + interpolated_mg_eg_score + interpolated_pawn_struct

    def initialize_scores(self, board: chess.Board) -> None:
        """
        Initialize values for starting position (works with custom starting FENs).
        Calculates material score, npm score, and evaluates piece positions.
        Evaluates piece positions using PSQT with interpolation between middlegame and endgame.
        Runs only once so not optimized for clarity.
        """
        self.material = np.int16(0)
        self.mg = np.int16(0)
        self.eg = np.int16(0)
        self.npm = np.uint16(0)
        self.pawn_struct = np.int8(0)  # Pawn structure score TODO: Check for overflow
        self.king_safety = np.int8(0)  # King safety score

        white_bishop_count = 0
        black_bishop_count = 0

        # Cache tables for faster lookups
        piece_values = PIECE_VALUES_STOCKFISH
        mg_tables = PSQT[MIDGAME]
        eg_tables = PSQT[ENDGAME]
        flip = FLIP
        bishop_bonus = BISHOP_PAIR_BONUS

        # Evaluate each piece type
        for square in chess.SQUARES:
            piece_type = board.piece_type_at(square)
            if piece_type:
                piece_color = board.color_at(square)

                # Update npm score
                if piece_type != chess.PAWN and piece_type != chess.KING:
                    self.npm += piece_values[piece_type]

                # Update material and position scores
                if piece_color: # White piece
                    self.material += piece_values[piece_type]
                    self.mg += mg_tables[piece_type][flip[square]] # type: ignore
                    self.eg += eg_tables[piece_type][flip[square]] # type: ignore
                    if piece_type == chess.BISHOP:
                        white_bishop_count += 1
                else: # Black piece
                    self.material -= piece_values[piece_type]
                    self.mg -= mg_tables[piece_type][square] # type: ignore
                    self.eg -= eg_tables[piece_type][square] # type: ignore
                    if piece_type == chess.BISHOP:
                        black_bishop_count += 1

        # Bishop pair bonus worth half a pawn
        if white_bishop_count >= 2:
            self.material += bishop_bonus
        if black_bishop_count >= 2:
            self.material -= bishop_bonus


        # Pawn structure
        file_masks = chess.BB_FILES

        # Process both colors at once with direct bitboard manipulation
        white_pawns = board.pieces_mask(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces_mask(chess.PAWN, chess.BLACK)

        # Evaluate isolated pawns for both colors in one pass
        for file in range(8):
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
                self.pawn_struct -= 20  # Isolated white pawn penalty
            if black_pawns_in_file > 0 and chess.popcount(black_pawns & adjacent_mask) == 0:
                self.pawn_struct += 20  # Isolated black pawn penalty

            # Check for doubled pawns
            if white_pawns_in_file > 1:
                self.pawn_struct -= 10  # Doubled white pawn penalty
            if black_pawns_in_file > 1:
                self.pawn_struct += 10  # Doubled black pawn penalty

    def updated(self, board: chess.Board, move: chess.Move) -> "Score":
        """
        Returns the updated material, midgame, endgame, and non-pawn material scores based on the move.
        Much faster than re-evaluating the entire board, even if only the leaf nodes are re-evaluated.
        Hard to understand, but worth it for performance.
        """
        _popcount = chess.popcount

        material, mg, eg, npm, pawn_struct, king_safety = self.material, self.mg, self.eg, self.npm, self.pawn_struct, self.king_safety

        from_square = move.from_square
        to_square = move.to_square
        promotion_piece_type: Optional[chess.PieceType] = move.promotion

        piece_type = board.piece_type_at(from_square) # ? Expensivish
        piece_color = board.turn
        color_multiplier = 1 if piece_color else -1

        # Cache tables for faster lookups
        piece_values: dict[int, int] = PIECE_VALUES_STOCKFISH
        mg_tables: list[Optional[np.ndarray]] = PSQT[MIDGAME]
        eg_tables: list[Optional[np.ndarray]] = PSQT[ENDGAME]
        flip = FLIP
        
        castling = False
        if piece_type == chess.PAWN: # Update pawn structure if moving a pawn
            pawns_before = board.pieces_mask(chess.PAWN, piece_color)
            pawns_after = pawns_before & ~(1 << from_square) # Remove moved pawn from pawns
            if promotion_piece_type: # If we are promoting, then we need to account for the dissapearance of the pawn
                file_masks = chess.BB_FILES
                file = from_square & 7

                pawns_in_file_after = _popcount(pawns_after & file_masks[file])

                if pawns_in_file_after == 1: # 2 pawns in file before
                    pawn_struct += color_multiplier * 10 # Remove doubled pawn penalty
                
                if pawns_in_file_after == 0:  # No pawns in file after move
                    left_pawns = _popcount(pawns_after & file_masks[file - 1]) if file > 0 else 0
                    right_pawns = _popcount(pawns_after & file_masks[file + 1]) if file < 7 else 0

                    # Check if left file is now isolated
                    if left_pawns >= 1 and (_popcount(pawns_after & file_masks[file - 2]) if file > 1 else 0) == 0:
                        pawn_struct -= color_multiplier * 20  # Add penalty for isolated left file
                    # Check if right file is now isolated
                    if right_pawns >= 1 and (_popcount(pawns_after & file_masks[file + 2]) if file < 6 else 0) == 0:
                        pawn_struct -= color_multiplier * 20  # Add penalty for isolated right file

            else:  # Not promoting
                pawns_after |= 1 << to_square # Add moved pawn to pawns

                file_masks = chess.BB_FILES
                file = from_square & 7
                to_file = to_square & 7

                pawns_in_file_after = _popcount(pawns_after & file_masks[file])

                if file != to_file and pawns_in_file_after == 1: # 2 pawns in file before
                    pawn_struct += color_multiplier * 10 # Remove doubled pawn penalty

                if to_file < file: # Move to left file (left, to_file, file, right)
                    left_pawns = _popcount(pawns_after & file_masks[to_file - 1]) if to_file > 0 else 0
                    pawns_in_to_file = _popcount(pawns_after & file_masks[to_file])
                    # pawns_in_file_after
                    right_pawns = _popcount(pawns_after & file_masks[file + 1]) if file < 7 else 0

                    if file != to_file and pawns_in_to_file == 2: # If now 2 pawns in file
                        pawn_struct -= color_multiplier * 10 # Add doubled pawn penalty

                    # If no longer isolated because of move (moved left, had no right pawns)
                    if pawns_in_to_file == 1 and right_pawns == 0:
                        pawn_struct += color_multiplier * 20 # Remove penalty

                    # Self isolating
                    if pawns_in_to_file >= 1 and left_pawns == 0 and pawns_in_file_after == 0:
                        pawn_struct -= color_multiplier * 20 # Add penalty

                    # Left was isolated previously (have left pawns, added a pawn, and no pawns left of left adj)
                    if left_pawns >= 1 and pawns_in_to_file == 1 and (_popcount(pawns_after & file_masks[to_file - 2]) if to_file > 1 else 0) == 0:
                        pawn_struct += color_multiplier * 20 # Remove penalty
                    if pawns_in_file_after == 0 and right_pawns >= 1 and (_popcount(pawns_after & file_masks[file + 2]) if file < 6 else 0) == 0: # Right adj is now isolated
                        pawn_struct -= color_multiplier * 20 # Add penalty
                        
                elif to_file > file: # Move to right file (left, file, to_file, right)
                    left_pawns = _popcount(pawns_after & file_masks[file - 1]) if file > 0 else 0
                    # pawns_in_file_after
                    pawns_in_to_file = _popcount(pawns_after & file_masks[to_file])
                    right_pawns = _popcount(pawns_after & file_masks[to_file + 1]) if to_file < 7 else 0

                    if file != to_file and pawns_in_to_file == 2: # If now 2 pawns in file
                        pawn_struct -= color_multiplier * 10 # Add doubled pawn penalty

                    # If no longer isolated because of move (moved right, had no left pawns)
                    if pawns_in_to_file == 1 and left_pawns == 0:
                        pawn_struct += color_multiplier * 20 # Remove penalty

                    # Self isolating
                    if pawns_in_to_file >= 1 and right_pawns == 0 and pawns_in_file_after == 0:
                        pawn_struct -= color_multiplier * 20 # Add penalty

                    if pawns_in_file_after == 0 and left_pawns >= 1 and (_popcount(pawns_after & file_masks[file - 2]) if file > 1 else 0) == 0: # Left adj is now isolated
                        pawn_struct -= color_multiplier * 20 # Add penalty
                    # Right was isolated previously (added a pawn, have right pawns, and no pawns right of right adj)
                    if right_pawns >= 1 and pawns_in_to_file == 1 and (_popcount(pawns_after & file_masks[to_file + 2]) if to_file < 6 else 0) == 0: # Right adj was isolated
                        pawn_struct += color_multiplier * 20 # Remove penalty

        elif piece_type == chess.KING: # Update rook scores if castling
            castle_info = CASTLING_UPDATES.get((from_square, to_square, piece_color))
            if castle_info:
                castling = True
                mg_rook_table = mg_tables[chess.ROOK]
                eg_rook_table = eg_tables[chess.ROOK]

                rook_from, rook_to = castle_info
                if piece_color: # Flip rook square for white
                    rook_from, rook_to = flip[rook_from], flip[rook_to]

                mg += color_multiplier * (mg_rook_table[rook_to] - mg_rook_table[rook_from]) # type: ignore
                eg += color_multiplier * (eg_rook_table[rook_to] - eg_rook_table[rook_from]) # type: ignore


        # Flip squares for white
        new_from_square, new_to_square = from_square, to_square
        if piece_color:
            new_from_square, new_to_square = flip[from_square], flip[to_square]

        # Update position scores for moving piece
        if promotion_piece_type: # Promotion
            # Update bishop pair bonus if pawn promoted to bishop
            if promotion_piece_type == chess.BISHOP:
                bishop_count_before = board.pieces_mask(chess.BISHOP, piece_color).bit_count()
                if bishop_count_before == 1: # If 2 bishops now, add bonus
                    material += color_multiplier * BISHOP_PAIR_BONUS

            npm += piece_values[promotion_piece_type]
            material += color_multiplier * (piece_values[promotion_piece_type] - piece_values[chess.PAWN])
            mg += color_multiplier * (mg_tables[promotion_piece_type][new_to_square] - mg_tables[chess.PAWN][new_from_square]) # type: ignore
            eg += color_multiplier * (eg_tables[promotion_piece_type][new_to_square] - eg_tables[chess.PAWN][new_from_square]) # type: ignore
        else: # Normal move
            mg_table = mg_tables[piece_type] # type: ignore
            eg_table = eg_tables[piece_type] # type: ignore
            mg += color_multiplier * (mg_table[new_to_square] - mg_table[new_from_square]) # ? Expensive
            eg += color_multiplier * (eg_table[new_to_square] - eg_table[new_from_square]) # ? Expensive (less)

        if castling: # Done if castling
            return Score(material, mg, eg, npm, pawn_struct, king_safety)


        # Handle captures
        captured_piece_type = board.piece_type_at(to_square) # ? Expensivish

        # Get en passant capture piece if applicable
        if not captured_piece_type and piece_type == chess.PAWN and board.is_en_passant(move):
            to_square -= color_multiplier * 8
            captured_piece_type = board.piece_type_at(from_square)

        if captured_piece_type: # Capture
            if captured_piece_type == chess.PAWN: # Capturing a pawn (update pawn structure)
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
                        pawn_struct += -color_multiplier * 20 # Remove penalty
                    if left_pawns >= 1 and (_popcount(enemy_pawns_after & file_masks[file - 2]) if file > 1 else 0) == 0: # Left adj isolated
                        pawn_struct -= -color_multiplier * 20 # Add penalty
                    if right_pawns >= 1 and (_popcount(enemy_pawns_after & file_masks[file + 2]) if file < 6 else 0) == 0: # Right adj isolated
                        pawn_struct -= -color_multiplier * 20 # Add penalty
                elif pawns_in_file_after == 1: # 2 pawns in file before
                    pawn_struct += -color_multiplier * 10 # Remove doubled pawn penalty

            else: # Capturing a piece other than a pawn
                # Update npm score
                npm -= piece_values[captured_piece_type]

                # Update bishop pair bonus if bishop captured
                if captured_piece_type == chess.BISHOP and board.pieces_mask(captured_piece_type, not piece_color).bit_count() == 2:
                    material -= -color_multiplier * BISHOP_PAIR_BONUS # If 2 bishops before, remove bonus

            if not piece_color: # Flip squares for white
                to_square = flip[to_square]

            # Remove captured piece from material and position scores
            material -= -color_multiplier * piece_values[captured_piece_type]
            mg -= -color_multiplier * mg_tables[captured_piece_type][to_square] # type: ignore
            eg -= -color_multiplier * eg_tables[captured_piece_type][to_square] # type: ignore

        return Score(material, mg, eg, npm, pawn_struct, king_safety) # ? Expensive


class ChessBot:
    __slots__ = ["game", "moves_checked", "transposition_table"]  # Optimization for fast lookups

    def __init__(self, game) -> None:
        self.game: "ChessGame" = game
        self.moves_checked: int = 0

        # Initialize transposition table with size in MB
        tt_entry_size = getsizeof(TTEntry(np.int8(0), np.int16(0), EXACT, chess.Move.from_uci("e2e4")), 64)
        self.transposition_table = LRU(int(TT_SIZE) * 1024 * 1024 // tt_entry_size)  # Initialize TT with size in MB

    def display_checking_move_arrow(self, move) -> None:
        """Display an arrow on the board for the move being checked."""
        self.game.checking_move = move
        self.game.display_board(self.game.last_move)  # Update display

    def evaluate_position(self, board: chess.Board, score: Score, tt_entry: Optional[TTEntry] = None, has_legal_moves=True) -> np.int16:
        """
        Evaluate the current position.
        Positive values favor white, negative values favor black.
        """
        if tt_entry:
            return tt_entry.value

        # Check expensive operations once
        if has_legal_moves: # TODO: Use pseduo legal and only push forward if legal?
            has_legal_moves = any(board.legal_moves) # ! REALLY SLOW

        # Evaluate game-ending conditions
        if not has_legal_moves:  # No legal moves
            if board.is_check():  # Checkmate
                return MIN_VALUE if board.turn else MAX_VALUE
            return np.int16(0)  # Stalemate
        elif board.is_insufficient_material(): # (semi-slow) Insufficient material for either side to win
            return np.int16(0)
        elif board.can_claim_fifty_moves(): # Avoid fifty move rule
            return np.int16(0)

        return score.calculate()

    # def quiescence(self, board: chess.Board, alpha, beta, depth):

    def ordered_moves_generator(self, board: chess.Board, tt_move: Optional[chess.Move]) -> Generator[chess.Move, None, None]:
        """Generate ordered moves for the current position."""
        # Cache functions for faster lookups
        _is_capture = board.is_capture
        _piece_type_at = board.piece_type_at

        # Yield transposition table move first
        if tt_move:
            yield tt_move

        # Cache tables for faster lookups
        piece_values = PIECE_VALUES_STOCKFISH

        color_multiplier = 1 if board.turn else -1

        # Sort remaining moves
        ordered_moves = []
        for move in board.legal_moves:
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
                    score += 10_000 + piece_values[victim_piece_type] - piece_values[attacker_piece_type] # type: ignore

                if move.promotion: # Promotion bonus
                    score += 1_000 + piece_values[move.promotion] - piece_values[chess.PAWN]

                # if board.gives_check(move): # Check bonus
                #     score += 100

                # # Center control bonus
                # if move.to_square in CENTER_SQUARES:
                #     score += 100

                ordered_moves.append((move, score))

        ordered_moves.sort(key=lambda x: x[1], reverse=True)

        for move_and_score in ordered_moves:
            yield move_and_score[0]


    def alpha_beta(self, board: chess.Board, depth: np.int8, alpha: np.int16, beta: np.int16, maximizing_player: bool, score: Score) -> tuple[np.int16, Optional[chess.Move]]:
        """
        Fail-soft alpha-beta search with transposition table.
        Scores are incrementally updated based on the move.
        Returns the best value and move for the current player.
        TODO, PV search, iterative deepening, quiescence search, killer moves, history heuristic, late move reduction, null move pruning
        """
        # Cache functions for faster lookups
        _push = board.push
        _pop = board.pop
        _score_updated = score.updated

        original_alpha, original_beta = alpha, beta

        # Lookup position in transposition table
        # key = zobrist_hash(board) # ! REALLY SLOW (probably because it is not incremental)
        key = board._transposition_key() # ? Much faster
        tt_entry: Optional[TTEntry] = self.transposition_table.get(key)

        # If position is in transposition table and depth is sufficient
        if tt_entry and tt_entry.depth >= depth:
            if tt_entry.flag == EXACT:
                return tt_entry.value, tt_entry.best_move
            elif tt_entry.flag == LOWERBOUND:
                alpha = np.int16(max(int(alpha), tt_entry.value))
            elif tt_entry.flag == UPPERBOUND:
                beta = np.int16(min(int(beta), tt_entry.value))

            if alpha >= beta:
                return tt_entry.value, tt_entry.best_move

        # Terminal node check
        if depth == 0:
            return self.evaluate_position(board, score, tt_entry), None

        tt_move = tt_entry.best_move if tt_entry else None
        best_move = None
        if maximizing_player:
            best_value = MIN_VALUE
            for move in self.ordered_moves_generator(board, tt_move):
                self.moves_checked += 1
                if CHECKING_MOVE_ARROW and depth >= RENDER_DEPTH:  # Display the root move
                    self.display_checking_move_arrow(move)

                updated_score: Score = _score_updated(board, move)

                _push(move)
                value: np.int16 = self.alpha_beta(board, np.int8(depth - 1), alpha, beta, False, updated_score)[0]
                _pop()

                if value > best_value: # Get new best value and move
                    best_value, best_move = value, move
                    alpha = np.int16(max(int(alpha), int(best_value))) # Get new alpha
                    if best_value >= beta:
                        break  # Beta cutoff (fail-high: opponent won't allow this position)

        else: # Minimizing player
            best_value = MAX_VALUE
            for move in self.ordered_moves_generator(board, tt_move):
                self.moves_checked += 1
                if CHECKING_MOVE_ARROW and depth >= RENDER_DEPTH:  # Display the root move
                    self.display_checking_move_arrow(move)

                updated_score: Score = _score_updated(board, move)

                _push(move)
                value: np.int16 = self.alpha_beta(board, np.int8(depth - 1), alpha, beta, True, updated_score)[0]
                _pop()

                if value < best_value: # Get new best value and move
                    best_value, best_move = value, move
                    beta = np.int16(min(int(beta), int(best_value))) # Get new beta
                    if best_value <= alpha:
                        break  # Alpha cutoff (fail-low: other positions are better)

        if best_move is None: # If no legal moves, evaluate position (best move is None if loop did iterate through legal moves)
            return self.evaluate_position(board, score, tt_entry, has_legal_moves=False), None

        # Store position in transposition table
        if best_value <= original_alpha: # TODO compare with original alpha and beta
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
        ordered_moves = list(self.ordered_moves_generator(board, None))
        subtree_count = len(ordered_moves)
        color_multipier = 1 if maximizing_player else -1

        original_score = self.game.score
        better_count = 0

        best_move = None
        best_value = None
        while beta - alpha >= 2 and better_count != 1:
            seperation_value = self.next_guess(alpha, beta, subtree_count)

            better_count = 0
            better = []
            for move in ordered_moves:
                self.moves_checked += 1
                if CHECKING_MOVE_ARROW and DEPTH >= RENDER_DEPTH:
                    self.display_checking_move_arrow(move)

                score = original_score.updated(board, move)

                board.push(move) # TODO -(sep_value+1)
                move_value = self.alpha_beta(board, DEPTH - 1, -(seperation_value), -(seperation_value-1), not maximizing_player, score)[0]
                board.pop()

                if color_multipier * move_value >= seperation_value:
                    better_count += 1
                    better.append(move)
                    best_move = move
                    best_value = move_value

            # Update alpha-beta range
            if better_count > 1:
                alpha = seperation_value

                # # Update number of sub-trees that exceeds seperation test value
                if subtree_count != better_count:
                    subtree_count = better_count
                    ordered_moves = better
            else:
                beta = seperation_value

        return best_value, best_move


    def get_move(self, board: chess.Board):
        """
        Main method to get the best move for the current player.
        """
        self.moves_checked = 0

        # Run minimax once with manual timing
        start_time = default_timer()

        alpha, beta = MIN_VALUE, MAX_VALUE

        best_value, best_move = self.alpha_beta(
            board,
            DEPTH,
            alpha,
            beta,
            board.turn,
            self.game.score) # type: ignore

        # best_value, best_move = self.best_node_search(board, alpha, beta, board.turn)

        print(f"Goal value: {best_value}")

        # assert best_move is not None, "No best move returned" # TODO remove when done testing
        if best_move is None:
            legal_moves = list(board.legal_moves)
            if len(legal_moves) == 1:
                best_move: chess.Move = legal_moves[0]
            else:
                print(f"{colors.RED}No best move returned{colors.RESET}")
                print(f"{colors.RED}Legal moves: {legal_moves}{colors.RESET}")

        self.game.score = self.game.score.updated(board, best_move)

        time_taken = default_timer() - start_time

        # TODO move print stuff into function
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

        # # Print cache statistics
        print(f"Transposition table: {colors.BOLD}{colors.MAGENTA}{transposition_table_entries:,}{colors.RESET} entries, "
            f"{colors.BOLD}{colors.CYAN}{tt_size_mb:.4f}{colors.RESET} MB")
        # print(f"Evaluation cache: {colors.BOLD}{colors.MAGENTA}{len(self.evaluation_cache):,}{colors.RESET} entries, "
        #       f"{colors.BOLD}{colors.CYAN}{eval_size_mb:.4f}{colors.RESET} MB")

        # Print the FEN
        print(f"FEN: {board.fen()}")

        return best_move
