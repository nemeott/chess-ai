"""Chess bot implementation."""

from collections.abc import Generator, Hashable
from typing import TYPE_CHECKING

import chess
import numpy as np
from chess import polyglot  # Polyglot for opening book
from llist import sllist  # For history  # ty:ignore[unresolved-import]
from lru import LRU  # For transposition table

if TYPE_CHECKING:
    from game import ChessGame  # Only import while type checking

import sys
from timeit import default_timer  # For debug timing

import colors  # Debug log colors
from constants import (
    CHECKING_MOVE_ARROW,
    DEPTH,
    MAX_VALUE,
    MIN_VALUE,
    OPENING_BOOK_PATH,
    PIECE_VALUES_STOCKFISH,
    RENDER_DEPTH,
    TT_SIZE,
)
from score import Score  # For position evaluation
from tt_entry import EXACT, LOWERBOUND, UPPERBOUND, TTEntry

np.seterr(all="raise")  # Raise warnings for all numpy errors


class ChessBot:
    """Class to represent the chess bot."""

    __slots__ = [
        "history",
        "killer_moves",
        "moves_checked",
        "opening_book",
        "quiescence_moves_checked",
        "root_halfmove_clock",
        "score",
        "transposition_table",
    ]  # Optimization for fast lookups

    def __init__(self, use_opening_book: bool = True) -> None:  # TODO: Add verbose/print option
        """Initialize the chess bot with the game instance.

        Also initializes the transposition table with size in MB.
        Initializes the opening book if using and valid.
        """
        self.score: Score = Score()
        self.moves_checked: int = 0
        self.quiescence_moves_checked: int = 0  # TODO: Quiescence hash table?

        # Initialize transposition table with size in MB
        tt_entry_size = TTEntry(np.int8(0), np.int16(0), EXACT, chess.Move.from_uci("e2e4")).__sizeof__()
        self.transposition_table = LRU(int(TT_SIZE) * 1024 * 1024 // tt_entry_size)  # Initialize TT with size in MB

        self.history: sllist = sllist()  # History table for detecting repetitions
        self.root_halfmove_clock: int = 0  # Root halfmove clock

        self.killer_moves: list[list[chess.Move]] = [[], []]  # Killer moves for alpha-beta pruning

        # Initialize opening book
        self.opening_book = None
        if use_opening_book and OPENING_BOOK_PATH:
            try:
                self.opening_book = polyglot.open_reader(OPENING_BOOK_PATH)
                print(f"{colors.GREEN}Opening book loaded successfully.{colors.RESET}")
            except Exception as e:  # noqa: BLE001
                print(f"{colors.RED}Error loading opening book: {e}{colors.RESET}")

    def set_score(self, score: Score) -> None:  # noqa: D102
        self.score = score

    def get_score(self) -> Score:  # noqa: D102
        return self.score

    def increment_moves_and_render_debug_arrow(self, depth: np.int8, move: chess.Move) -> None:
        """Increment the move counter and render the debug arrow if enabled."""
        self.moves_checked += 1
        # if CHECKING_MOVE_ARROW and depth >= RENDER_DEPTH: # Display the root move
        #     self.game.arrow_move = move
        #     self.game.display_board(self.game.last_move) # Update display

    def is_repetition(self, board: chess.Board, key: Hashable, depth: np.int8) -> bool:
        """Check if the current position is a repetition.

        The first move in history at this point is the position from the opponent's last move.
        """
        # Check for repetitions
        halfmove_clock = board.halfmove_clock
        if halfmove_clock >= 4:  # Three-fold repetition possible, need to check
            root_halfmove_clock = self.root_halfmove_clock
            repetitions = 0
            for i, move_key in enumerate(self.history):
                # Only check our moves (history starts at the position from opponent's last move)
                if i % 2 == 1 and move_key == key:
                    return True

                    # # TODO: Figure out why actual repetition logic is not preventing repetitions
                    # repetitions += 1

                    # if repetitions == 2: # This is the 3rd repetition
                    #     return True
                    # # ply > root_ply + 2
                    # elif repetitions == 1 and DEPTH - depth > 2: # Prevent treating moves close to root as draws
                    #     return True

                if i >= halfmove_clock:  # Reached the last irreversible move
                    break

        return False

    def ordered_moves_generator(
        self,
        board: chess.Board,
        tt_move: chess.Move | None,
    ) -> Generator[chess.Move, None, None]:
        """Generate ordered moves for the current position.

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

        color_multiplier = 1 if board.turn else -1  # 1 for white, -1 for black

        # Sort remaining moves
        ordered_moves = []
        for move in board.generate_legal_moves():  # ! REALLY SLOW
            if not tt_move or move != tt_move:  # Skip TT move since already yielded
                score = 0

                # Capturing a piece bonus (MVV/LVA - Most Valuable Victim/Least Valuable Attacker)
                if _is_capture(move):
                    victim_piece_type = _piece_type_at(move.to_square)
                    attacker_piece_type = _piece_type_at(move.from_square)

                    # Handle en passant captures
                    if not victim_piece_type:  # Implied en passant capture since no piece at to_square and pawn moving
                        victim_piece_type = _piece_type_at(move.to_square - (color_multiplier * 8))
                        score += 5  # Small bonus for en passant captures

                    # TODO: Sort good vs bad captures
                    # Prioritize capturing higher value pieces using lower value pieces
                    score += 10_000 + int(
                        _piece_values[victim_piece_type]  # ty:ignore[invalid-argument-type]
                        - _piece_values[attacker_piece_type]  # ty:ignore[invalid-argument-type]  # noqa: COM812
                    )

                # TODO: Killer moves
                # if move in self.killer_moves:
                #     score += 1_000

                if move.promotion:  # Promotion bonus
                    score += 100 + _piece_values[move.promotion] - _piece_values[chess.PAWN]

                # if score == 0 and board.gives_check(move): # ! SLOW
                #     score += 100

                ordered_moves.append((move, score))

        ordered_moves.sort(key=lambda x: x[1], reverse=True)

        for move_and_score in ordered_moves:
            yield move_and_score[0]

    # @staticmethod
    # def worth_capturing(board: chess.Board, move: chess.Move) -> bool:
    #     if move.promotion:
    #         return True

    #     attacker_piece_type = board.piece_type_at(move.from_square)
    #     victim_piece_type = board.piece_type_at(move.to_square)

    #     # Handle en passant captures
    #     if not victim_piece_type: # Implied en passant capture, the move is a capture
    #         color_multiplier = 1 if board.turn else -1 # 1 for white, -1 for black
    #         victim_piece_type = board.piece_type_at(move.to_square - (color_multiplier * 8))

    #     return PIECE_VALUES_STOCKFISH[victim_piece_type] - PIECE_VALUES_STOCKFISH[attacker_piece_type] >= 0

    # def quiescence(self, board: chess.Board, depth, alpha, beta, score) -> np.int16:
    #     """
    #     Quiescence search to avoid horizon effect.
    #     Searches only captures until a quiet position is reached.
    #     """
    #     self.quiescence_moves_checked += 1

    #     # Evaluate position (lazy evaluation)
    #     stand_pat = score.calculate()

    #     if depth == 0:
    #         return stand_pat

    #     if stand_pat >= beta: # Beta cutoff
    #         return beta # TODO: Test vs return stand_pat
    #     if alpha < stand_pat: # Update alpha if stand pat is better
    #         alpha = stand_pat

    #     # Cache functions for faster lookups
    #     _push = board.push
    #     _pop = board.pop

    #     for move in board.generate_legal_captures():
    #         if not self.worth_capturing(board, move): # Skip if capture is not worth it
    #             continue

    #         updated_score: Score = score.updated(board, move)

    #         _push(move)
    #         value = self.quiescence(board, depth - 1, -beta, -alpha, updated_score)
    #         _pop()

    #         if value >= beta: # Beta cutoff
    #             return beta
    #         if value > alpha: # Update alpha if value is better
    #             alpha = value

    #     return alpha # Return the best value found

    # def prune_null_moves
    # --- Null Move Pruning ---
    # Allow null move pruning if:
    # 1. Last move was not a null move
    # 2. Depth is greater than or equal to 3
    # 3. The player has non-pawn pieces (avoid zugzwang)
    # 4. The score is not too low
    # 5. The position is not in check
    # if allow_null_move and depth >= 3:
    #     maximizing_player = True if color_multiplier == 1 else False
    #     bitboard: chess.Bitboard = board.occupied_co[maximizing_player] # Get bitboard for current player
    #     bitboard = bitboard & ~board.pawns & ~board.kings # Remove pawns and kings from bitboard
    #     if bitboard > 0: # If there are non-pawn pieces (avoid zugzwang)
    #         eval = np.int16(color_multiplier) * score.calculate()
    #         if eval >= gamma and not board.is_check(): # If not in check
    #             R = 3 if (depth > 3 and score.npm < 1_500) else 2 # Reduction factor

    #             _push(chess.Move.null()) # Make null move (skip turn)
    #             null_value: np.int16 = -_mt_negamax(board, depth - R - 1, -gamma,
    #                                                     -color_multiplier, score, allow_null_move=False)[0]
    #             _pop() # Undo null move

    #             if null_value >= gamma: # If null move causes beta cutoff, prune this subtree
    #                 return null_value, None

    def mt_negamax(  # noqa: C901, PLR0911, PLR0912
        self,
        board: chess.Board,
        depth: np.int8,
        gamma: np.int16,
        color_multiplier: np.int16,
        score: Score,
        allow_null_move: bool = True,
    ) -> tuple[np.int16, chess.Move | None]:
        """Memory-enhanced Test negamax search algorithm proposed by Jan-Jaap van Horssen.

        Uses a null window search to find the best move.
        Gamma replaces beta, and alpha is replaced with beta - 1.
        Scores are incrementally updated based on the move.
        Returns the best value and move for the current player.
        """
        key: Hashable = board._transposition_key()  # ? Much faster than python-chess's zobrist hashing  # noqa: SLF001

        # Evaluate game-ending conditions
        best_move = next(board.generate_legal_moves(), None)  # ! SLOW
        if not best_move:  # No legal moves
            if board.is_check():  # Checkmate
                return np.int16(MIN_VALUE + (DEPTH - depth)), None  # Subtract depth to encourage faster mate
            return np.int16(0), None  # Stalemate
        # Avoid insufficient material, fifty move rule, threfold repetition
        if board.is_insufficient_material() or board.can_claim_fifty_moves() or self.is_repetition(board, key, depth):
            return np.int16(0), None

        # Terminal node check
        if depth == 0:
            value = color_multiplier * score.calculate()
            return value, None  # No move to return
            # return self.quiescence(board, 3, alpha, beta, score), None # No move to return

        # Lookup position in transposition table
        tt_entry: TTEntry | None = self.transposition_table.get(key)

        # If position is in transposition table and depth is sufficient
        tt_move = None
        if tt_entry:
            tt_move = tt_entry.best_move  # Save best TT move (use later in move ordering)
            if tt_entry.depth >= depth:
                if tt_entry.flag == LOWERBOUND and tt_entry.value >= gamma:
                    return tt_entry.value, tt_move
                if tt_entry.flag == UPPERBOUND and tt_entry.value < gamma:
                    return tt_entry.value, tt_move  # TODO: Test returning different stuff

        # Cache functions for faster lookups
        _mt_negamax = self.mt_negamax
        _push = board.push
        _pop = board.pop
        _score_updated = score.updated
        _increment_moves_and_render_arrow = self.increment_moves_and_render_debug_arrow

        self.history.appendleft(key)  # Add position to history

        best_value: np.int16 = MIN_VALUE
        for move in self.ordered_moves_generator(board, tt_move):
            _increment_moves_and_render_arrow(depth, move)  # TODO: Move increment outside to start of alpha-beta

            updated_score: Score = _score_updated(board, move)
            _push(move)
            value = -_mt_negamax(board, depth - 1, -gamma + 1, -color_multiplier, updated_score)[0]
            _pop()

            if value > best_value:  # Get new best value and move
                best_value, best_move = value, move
                if best_value >= gamma:
                    break  # Beta cutoff (fail-high: opponent won't allow this position)

        self.history.popleft()  # Remove position from history

        # Store position in transposition table according to Jan-Jaap van Horssen's algorithm
        # There are three viable options for the new TT move in function MT:
        # tt.store(board, g, g < gamma ? UPPER : LOWER, depth, <NEW_TT_MOVE>);
        #     Comparisons made to negamax
        #     OLD (50.1%): <NEW_TT_MOVE> := move (inconsistent start, middle same, end different but fine) (least moves checked)  # noqa: E501
        #     NEW (49.6%): <NEW_TT_MOVE> := g >= gamma ? move : NO_MOVE (inconsistent start, middle same, end different but fine)  # noqa: E501
        #     HIGH (50.2%): <NEW_TT_MOVE> := g >= gamma ? move : ttMove (inconsistent start, rest same) (2nd least moves checked)  # noqa: E501
        #     All start the same except for the end
        if best_value < gamma:  # noqa: SIM108
            flag = UPPERBOUND
            # best_move = tt_move # Uncomment for HIGH
            # best_move = None # Uncomment for NEW (requires fix in MTD)
        else:  # best_value >= gamma
            flag = LOWERBOUND

        self.transposition_table[key] = TTEntry(depth, best_value, flag, best_move)

        return best_value, best_move

        # # --- Null Move Pruning ---
        # # Allow null move pruning if:
        # # 1. Last move was not a null move
        # # 2. Depth is greater than or equal to 3
        # # 3. The player has non-pawn pieces (avoid zugzwang)
        # # 4. The score is not too low
        # # 5. The position is not in check
        # if allow_null_move and depth >= 3:
        #     maximizing_player = True if color_multiplier == 1 else False
        #     bitboard: chess.Bitboard = board.occupied_co[maximizing_player] # Get bitboard for current player
        #     bitboard = bitboard & ~board.pawns & ~board.kings # Remove pawns and kings from bitboard
        #     if bitboard > 0: # If there are non-pawn pieces (avoid zugzwang)
        #         value = score.calculate()
        #         if (maximizing_player and value >= beta) or (not maximizing_player and value <= alpha):
        #             if not board.is_check(): # If not in check
        #                 R = 3 if (depth > 3 and score.npm < 1_500) else 2 # Reduction factor

        #                 _push(chess.Move.null()) # Make null move (skip turn)
        #                 null_value: np.int16 = self.negamax_alpha_beta(board, depth - R - 1, -beta, -beta + 1,

    #                                                                -color_multiplier, score, allow_null_move=False)[0]
    #                 _pop() # Undo null move

    #                 # If null move causes beta cutoff, prune this subtree
    #                 if null_value >= beta:
    #                     return null_value, None

    def negamax_alpha_beta(  # noqa: C901, PLR0911, PLR0912
        self,
        board: chess.Board,
        depth: np.int8,
        alpha: np.int16,
        beta: np.int16,
        color_multiplier: np.int16,
        score: Score,
        allow_null_move: bool = True,
    ) -> tuple[np.int16, chess.Move | None]:
        """Negamax search algorithm with alpha-beta pruning and transposition table.

        Scores are incrementally updated based on the move.
        Returns the best value and move for the current player.
        """
        key: Hashable = board._transposition_key()  # ? Much faster  # noqa: SLF001

        # Evaluate game-ending conditions
        best_move = next(board.generate_legal_moves(), None)  # Get first move
        if not best_move:  # No legal moves
            if board.is_check():  # Checkmate
                return np.int16(MIN_VALUE + (DEPTH - depth)), None  # Subtract depth to encourage faster mate
            return np.int16(0), None  # Stalemate
        # Avoid insufficient material, fifty move rule, threfold repetition
        if board.is_insufficient_material() or board.can_claim_fifty_moves() or self.is_repetition(board, key, depth):
            return np.int16(0), None

        # Terminal node check
        if depth == 0:
            value = color_multiplier * score.calculate()
            # self.transposition_table[key] = TTEntry(depth, value, EXACT, None) # TODO: Test
            return value, None  # No move to return
            # return self.quiescence(board, 3, alpha, beta, score), None # No move to return

        # Lookup position in transposition table
        tt_entry: TTEntry | None = self.transposition_table.get(key)

        # If position is in transposition table and depth is sufficient
        tt_move = None
        if tt_entry:
            tt_move = tt_entry.best_move
            if tt_entry.depth >= depth:
                if tt_entry.flag == EXACT:
                    return tt_entry.value, tt_move
                if tt_entry.flag == LOWERBOUND and tt_entry.value >= beta:
                    return tt_entry.value, tt_move
                if tt_entry.flag == UPPERBOUND and tt_entry.value <= alpha:
                    return tt_entry.value, tt_move

        # Cache functions for faster lookups
        _push = board.push
        _pop = board.pop
        _score_updated = score.updated
        _increment_moves_and_render_arrow = self.increment_moves_and_render_debug_arrow

        original_alpha = alpha

        self.history.appendleft(key)  # Add position to history

        best_value: np.int16 = MIN_VALUE
        for move in self.ordered_moves_generator(board, tt_move):
            _increment_moves_and_render_arrow(depth, move)  # TODO: Move increment outside to start of alpha-beta

            updated_score: Score = _score_updated(board, move)
            _push(move)
            value: np.int16 = -self.negamax_alpha_beta(
                board,
                depth - 1,
                -beta,
                -alpha,
                -color_multiplier,
                updated_score,
            )[0]
            _pop()

            if value > best_value:  # Get new best value and move
                best_value, best_move = value, move
                alpha = np.int16(max(int(alpha), int(best_value)))  # Get new alpha
                if alpha >= beta:
                    break  # Beta cutoff (fail-high: opponent won't allow this position)

        self.history.popleft()  # Remove position from history

        # Store position in transposition table
        if best_value <= original_alpha:
            flag = UPPERBOUND
        elif best_value >= beta:
            flag = LOWERBOUND
        else:
            flag = EXACT

        self.transposition_table[key] = TTEntry(depth, best_value, flag, best_move)

        return best_value, best_move

        # # --- Null Move Pruning ---
        # # Allow null move pruning if:
        # # 1. Last move was not a null move
        # # 2. Depth is greater than or equal to 3
        # # 3. The player has non-pawn pieces (avoid zugzwang)
        # # 4. The score is not too low
        # # 5. The position is not in check
        # if allow_null_move and depth >= 3:
        #     bitboard: chess.Bitboard = board.occupied_co[maximizing_player] # Get bitboard for current player
        #     bitboard = bitboard & ~board.pawns & ~board.kings # Remove pawns and kings from bitboard
        #     if bitboard > 0: # If there are non-pawn pieces (avoid zugzwang)
        #         value = score.calculate()
        #         if (maximizing_player and value >= beta) or (not maximizing_player and value <= alpha):
        #             if not board.is_check(): # If not in check
        #                 R = 3 if (depth > 3 and score.npm < 1_500) else 2 # Reduction factor

        #                 _push(chess.Move.null()) # Make null move (skip turn)
        #                 if maximizing_player:
        #                     null_value: np.int16 = self.alpha_beta(board, np.int8(
        #                         depth - R - 1), -beta, -beta + 1, not maximizing_player, score, allow_null_move=False)[0]  # noqa: E501
        #                 else:
        #                     null_value: np.int16 = self.alpha_beta(board, np.int8(
        #                         depth - R - 1), -alpha - 1, -alpha, not maximizing_player, score, allow_null_move=False)[0]  # noqa: E501

        #                 _pop() # Undo null move

        #                 # If null move causes beta cutoff, prune this subtree
        #                 if maximizing_player and null_value >= beta:
        #                     return null_value, None
        #                 elif not maximizing_player and null_value <= alpha:
        #                     return null_value, None

    # TODO: WIP ---------------------------------------------

    # def next_guess(self, alpha, beta, subtree_count) -> float:
    #     return alpha + (beta - alpha) * (subtree_count - 1) / subtree_count

    # def best_node_search(self, board: chess.Board, alpha, beta, maximizing_player: bool):
    #     """
    #     Experimental best node search (fuzzified game search) algorithm based on the paper by Dmitrijs Rutko.
    #     Uses the next guess function to return the separation value for the next iteration.
    #     """
    #     alpha, beta = float(alpha), float(beta)

    #     ordered_moves = list(self.ordered_moves_generator(board, None))
    #     subtree_count = len(ordered_moves)
    #     color_multiplier = 1 if maximizing_player else -1

    #     original_score = self.score
    #     better_count = 0

    #     best_move = None
    #     best_value = None
    #     while beta - alpha >= 2 and better_count != 1:
    #         separation_value = self.next_guess(alpha, beta, subtree_count)

    #         better_count = 0
    #         better = []
    #         for move in ordered_moves:
    #             self.increment_moves_and_render_debug_arrow(DEPTH, move)

    #             score = original_score.updated(board, move)
    #             board.push(move)
    #             move_value = self.alpha_beta(board,
    #                                          DEPTH - 1,
    #                                          -(separation_value),
    #                                          -(separation_value - 1),
    #                                          not maximizing_player,
    #                                          score)[0]
    #             board.pop()

    #             if color_multiplier * int(move_value) >= separation_value:
    #                 better_count += 1
    #                 better.append(move)
    #                 best_move = move
    #                 best_value = move_value

    #                 # expected_quality = 1 / ((better_count / self.moves_checked) * (subtree_count**int(DEPTH)))
    #                 # if expected_quality > 0.95:
    #                 #     return best_value, best_move

    #         # Update alpha-beta range
    #         if better_count > 1:
    #             alpha = separation_value

    #             # Update number of sub-trees that exceeds separation test value
    #             if subtree_count != better_count:
    #                 subtree_count = better_count
    #                 ordered_moves = better
    #         else:
    #             beta = separation_value

    #     return best_value, best_move

    def iterative_deepening_mtd_fix_driver(self, board: chess.Board) -> tuple[np.int16, chess.Move | None]:
        """Iterative deepening driver for MTD(f) search."""  # noqa: D401
        key: Hashable = board._transposition_key()  # ? Much faster than python-chess's zobrist hashing  # noqa: SLF001
        color_multiplier = np.int16(1) if board.turn else np.int16(-1)  # 1 for white, -1 for black

        first_guess, best_move = np.int16(0), None
        for depth in range(1, DEPTH + 1):  # TODO: Test 0
            # first_guess, best_move = self.mtd_fix(board, first_guess, np.int8(depth), color_multiplier)
            first_guess, best_move = self.mtd_safe_fix(board, first_guess, np.int8(depth), color_multiplier, key)

        return first_guess, best_move

    def mtd_safe_fix(
        self,
        board: chess.Board,
        first_guess: np.int16,
        depth: np.int8,
        color_multiplier: np.int16,
        key: Hashable,
    ) -> tuple[np.int16, chess.Move | None]:
        """MTD(f) search algorithm enhanced with a fix proposed by Jan-Jaap van Horssen in Handling Search Inconsistencies in MTD(f).

        The algorithm uses a binary search to find the best move.
        The results are narrowed down, using a null window search.
        The best move is chosen based off of the SAFE protocol described in the paper Move selection in MTD(f )
        The fix is to revert to the previous best move, if the last pass failed low and the best move has changed.
        This prevents the algorithm from returning a move that is worse than the previous best move.
        It also prevents us from having to clear the transposition table after each iteration.
        """  # noqa: E501
        """
        TODO:
            During each call to the Memory Enhanced Test
            algorithm, establish an initial principal variation by
            disabling null move pruning until the first leaf node
            has been evaluated. This is shown in Figure 3.
            2. If a node along the principal variation, fails high
            after the first move fails low, the principal variation
            has changed. It is possible that this new principal
            variation contains a null move. Figure 4 shows such
            a case.
            3. Perform a re-search from the node where the fail
            high has occurred. Null move pruning must be
            disabled for the re-search. Figure 5 shows the
            resulting search tree after a re-search has removed
            the null move from the principal variation.
        """

        """
        TODO: Inconsistent at start (has to warm up some?)
        """

        gamma = 0  # Overwritten in loop
        g = first_guess
        lower_bound, upper_bound = MIN_VALUE, MAX_VALUE

        prev_best_guess, prev_best_move = None, None
        best_move = None
        while lower_bound < upper_bound:
            prev_best_guess, prev_best_move = g, best_move

            gamma = g + 1 if g == lower_bound else g

            g, best_move = self.mt_negamax(board, depth, gamma, color_multiplier, self.score)

            if g < gamma:
                upper_bound = g
            else:  # g >= gamma
                lower_bound = g

                tt_entry: TTEntry | None = self.transposition_table.get(key)
                if tt_entry and tt_entry.best_move:
                    best_move = tt_entry.best_move

        # Needed for the NEW move transposition table replacement scheme
        if (
            g < gamma and best_move != prev_best_move
        ):  # If last pass failed low and best move changed, replace with previous
            g, best_move = prev_best_guess, prev_best_move

        return g, best_move  # ty:ignore[invalid-return-type]

    def mtd_fix(
        self,
        board: chess.Board,
        first_guess: np.int16,
        depth: np.int8,
        color_multiplier: np.int16,
    ) -> tuple[np.int16, chess.Move | None]:
        """MTD(f) search algorithm enhanced with a fix proposed by Jan-Jaap van Horssen.

        The algorithm uses a binary search to find the best move.
        The results are narrowed down, using a null window search.
        The fix is to revert to the previous best move, if the final value is lower than the guess and the best move has changed.
        This prevents the algorithm from returning a move that is worse than the previous best move.
        It also prevents us from having to clear the transposition table after each iteration.
        """  # noqa: E501
        guess = beta = first_guess
        upper_bound, lower_bound = MAX_VALUE, MIN_VALUE

        prev_best_guess, prev_best_move = None, None
        best_move = None
        while lower_bound < upper_bound:
            prev_best_guess, prev_best_move = guess, best_move

            beta = np.int16(max(int(guess), int(lower_bound) + 1))

            guess, best_move = self.negamax_alpha_beta(board, depth, beta - 1, beta, color_multiplier, self.score)
            if guess < beta:
                upper_bound = guess
            else:
                lower_bound = guess

        # The fix: if final value is lower than guess and best move changed, revert to previous
        if guess < beta and best_move != prev_best_move:
            guess, best_move = prev_best_guess, prev_best_move

        return guess, best_move  # ty:ignore[invalid-return-type]

    def mtd_f(
        self,
        board: chess.Board,
        first_guess: np.int16,
        color_multiplier: np.int16,
    ) -> tuple[np.int16, chess.Move | None]:
        """MTD(f) search algorithm.

        The algorithm uses a binary search to find the best move.
        The results are narrowed down, using a null window search.
        """
        guess = first_guess
        upper_bound, lower_bound = MAX_VALUE, MIN_VALUE

        best_move = None
        while lower_bound < upper_bound:
            beta = np.int16(max(int(guess), int(lower_bound) + 1))

            guess, best_move = self.negamax_alpha_beta(board, DEPTH, beta - 1, beta, color_multiplier, self.score)
            if guess < beta:
                upper_bound = guess
            else:
                lower_bound = guess

        return guess, best_move

    def get_move(self, board: chess.Board) -> chess.Move | None:
        """Return the best move for the current player."""
        key = board._transposition_key()  # noqa: SLF001

        # Set the root halfmove clock
        self.root_halfmove_clock = board.halfmove_clock

        self.moves_checked = 0

        alpha, beta = MIN_VALUE, MAX_VALUE

        time_taken: float = 0.0
        best_move: chess.Move | None = chess.Move.null()
        if self.opening_book:
            book_entry = self.opening_book.get(board)  # Get the best book move
            if book_entry:  # Use opening book move
                best_move = book_entry.move

        if not best_move:  # No book move found, use alpha-beta search
            start_time: float = default_timer()  # Start timer

            color_multiplier = np.int16(1) if board.turn else np.int16(-1)  # 1 for white, -1 for black
            # best_value, best_move = self.best_node_search(board, alpha, beta, board.turn)

            # best_value, best_move = self.negamax_alpha_beta(board, DEPTH, alpha, beta, color_multiplier, self.score)
            best_value, best_move = self.iterative_deepening_mtd_fix_driver(board)

            best_value *= color_multiplier

            time_taken = default_timer() - start_time  # Stop timer

            # print(f"Goal value: {best_value}")

        if best_move is None:
            legal_moves = list(board.generate_legal_moves())
            if len(legal_moves) > 0:
                print(f"{colors.RED}No best move returned, using first legal move{colors.RESET}")
                print(f"Had {len(legal_moves)} legal moves")
                print(f"Legal moves: {legal_moves}")
                print(f"FEN: {board.fen()}")
                best_move = legal_moves[0]
            else:
                print(f"{colors.RED}No best move returned{colors.RESET}")
                print(f"{colors.RED}Legal moves: {legal_moves}{colors.RESET}")
                sys.exit()

        self.score = self.score.updated(board, best_move)  # ty:ignore[invalid-argument-type]

        # Add start pos and end pos to history
        self.history.appendleft(key)  # Add position to history
        board.push(best_move)  # ty:ignore[invalid-argument-type]
        # print(self.is_repetition(board, board._transposition_key(), DEPTH))
        self.history.appendleft(board._transposition_key())  # Add position to history  # noqa: SLF001
        board.pop()

        # self.print_stats(board, time_taken)

        return best_move

    def print_stats(self, board: chess.Board, time_taken: float) -> None:
        """Print statistics about the search.

        Prints the number of moves checked, time taken, moves per second, and transposition table size.
        """
        # Moves checked over time taken
        time_per_move = time_taken / self.moves_checked if self.moves_checked > 0 else 0
        moves_per_second = 1 / time_per_move if time_per_move > 0 else 0
        # print(f"Moves/Time: {colors.BOLD}{colors.get_moves_color(self.moves_checked)}{self.moves_checked:,}{colors.RESET} / "
        #       f"{colors.BOLD}{colors.get_move_time_color(time_taken)}{time_taken:.2f}{colors.RESET} s = "
        #       f"{colors.BOLD}{colors.CYAN}{time_per_move * 1000:.4f}{colors.RESET} ms/M, "
        #       f"{colors.BOLD}{colors.CYAN}{moves_per_second:,.0f}{colors.RESET} M/s")
        print(
            f"Moves checked: {colors.BOLD}{colors.get_moves_color(self.moves_checked)}{self.moves_checked:,}{colors.RESET}",  # noqa: E501
        )

        # Calculate memory usage more accurately
        tt_entry_size = TTEntry(np.int8(0), np.int16(0), EXACT, chess.Move.from_uci("e2e4")).__sizeof__()
        transposition_table_entries = len(self.transposition_table)
        tt_size_mb = transposition_table_entries * tt_entry_size / (1024 * 1024)

        # Print cache statistics
        print(
            f"Transposition table: {colors.BOLD}{colors.MAGENTA}{transposition_table_entries:,}{colors.RESET} entries, "
            f"{colors.BOLD}{colors.CYAN}{tt_size_mb:.4f}{colors.RESET} MB",
        )

        # Print the FEN
        print(f"FEN: {board.fen()}")
