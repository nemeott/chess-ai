import chess

import numpy as np
from numba import jit # (njit not needed since default is nopython since numba 0.59.0)

from dataclasses import dataclass
from typing import Optional

from constants import PIECE_VALUES_STOCKFISH, BISHOP_PAIR_BONUS, DOUBLED_PAWN_PENALTY, ISOLATED_PAWN_PENALTY, FLIP, MIDGAME, ENDGAME, PSQT, CASTLING_UPDATES, NPM_SCALAR


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
        return Score._numba_calculate(self.material, self.mg, self.eg, self.npm, self.pawn_struct, self.king_safety)

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

    def updated(self, board: chess.Board, move: chess.Move) -> "Score": # TODO: Interpolate between mg and eg material scores
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
