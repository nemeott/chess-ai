import chess
from chess import polyglot # Polyglot for opening book
# from chess.polyglot import zobrist_hash # Built-in Zobrist hashing  TODO implement incremental hashing
import numpy as np

from lru import LRU # For TT and history tables
from sys import getsizeof # For memory usage calculations

from dataclasses import dataclass
from typing_extensions import TypeAlias # For flags
from typing import Generator, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from game import ChessGame # Only import while type checking

from constants import DEPTH, MAX_VALUE, MIN_VALUE, CHECKING_MOVE_ARROW, RENDER_DEPTH, TT_SIZE, PIECE_VALUES_STOCKFISH, OPENING_BOOK_PATH
from score import Score # For position evaluation
import colors # Debug log colors

from timeit import default_timer # For debug timing


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


class ChessBot:
    """
    Class to represent the chess bot.
    """
    __slots__ = ["game", "moves_checked", "quiescence_moves_checked",
                 "transposition_table", "opening_book", "history"] # Optimization for fast lookups

    def __init__(self, game, use_opening_book: bool = True) -> None:
        """
        Initialize the chess bot with the game instance.
        Also initializes the transposition table with size in MB.
        """
        self.game: "ChessGame" = game
        self.moves_checked: int = 0
        self.quiescence_moves_checked: int = 0

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

        self.history: set = set()

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
                    if not victim_piece_type: # Implied en passant capture since no piece at to_square and pawn moving
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

    #     return PIECE_VALUES_STOCKFISH[victim_piece_type] - PIECE_VALUES_STOCKFISH[attacker_piece_type] >= 0 # type: ignore # nopep8

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
        tt_entry: Optional[TTEntry] = self.transposition_table.get(key)

        # Add position to history
        # position_in_history = key in self.history # If the position is already in history
        # self.history.add(key) # Add position to history

        # If position is in transposition table and depth is sufficient
        tt_move = None
        if tt_entry and tt_entry.depth >= depth:
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
            self.transposition_table[key] = TTEntry(depth, value, EXACT, None) # ? Slowish
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

        if best_move is None: # If no legal moves (fall-through), evaluate position
            return self.evaluate_position(board, score, tt_entry, has_legal_moves=False), None

        # Remove position from history if not already in history
        # if not position_in_history:
        #     self.history.remove(key)

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

    def next_guess(self, alpha, beta, subtree_count) -> float:
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
                                             -(separation_value),
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

    def iterative_deepening_mtd_fix(self, board: chess.Board) -> tuple[np.int16, Optional[chess.Move]]:
        """
        Iterative deepening driver for MTD(f) search.
        TODO: Investigate different ways to calculate the initial first guess.
            An initial guess of 0 results in the best goal value being returned but runs into instability at the end.
            An initial guess of the current score rarely doesn't return the best goal value, but *does return the best move.
                *More testing needed to confirm this.
        """
        # first_guess, best_move = np.int16(0), None
        first_guess, best_move = self.game.score.calculate(), None
        for depth in range(1, DEPTH + 1):
            first_guess, best_move = self.mtd_fix(board, first_guess) # TODO: Try without

        return first_guess, best_move

    def mtd_fix(self, board: chess.Board, first_guess: np.int16) -> tuple[np.int16, Optional[chess.Move]]:
        """
        MTD(f) search algorithm enhanced with a fix proposed by Jan-Jaap van Horssen.
        The algorithm uses a binary search to find the best move.
        The results are narrowed down, using a null window search.
        The fix is to revert to the previous best move, if the final value is lower than the guess and the best move has changed.
        This prevents the algorithm from returning a move that is worse than the previous best move.
        It also prevents us from having to clear the transposition table after each iteration.
        """
        guess = beta = first_guess
        upper_bound, lower_bound = MAX_VALUE, MIN_VALUE

        prev_best_guess, prev_best_move = None, None
        best_move = None
        while lower_bound < upper_bound:
            prev_best_guess, prev_best_move = guess, best_move

            beta = max(int(guess), int(lower_bound) + 1)

            guess, best_move = self.alpha_beta(board, DEPTH, beta - 1, beta,
                                               board.turn, self.game.score, allow_null_move=False)
            if guess < beta:
                upper_bound = guess
            else:
                lower_bound = guess

        # The fix: if final value is lower than guess and best move changed, revert to previous
        if guess < beta and best_move != prev_best_move:
            guess, best_move = prev_best_guess, prev_best_move

        return guess, best_move # type: ignore

    def mtd_f(self, board: chess.Board, first_guess: np.int16) -> tuple[np.int16, Optional[chess.Move]]:
        """
        MTD(f) search algorithm.
        The algorithm uses a binary search to find the best move.
        The results are narrowed down, using a null window search.
        """
        guess = first_guess
        upper_bound, lower_bound = MAX_VALUE, MIN_VALUE

        best_move = None
        while lower_bound < upper_bound:
            beta = max(int(guess), int(lower_bound) + 1)

            guess, best_move = self.alpha_beta(board, DEPTH, beta - 1, beta, board.turn, self.game.score)
            if guess < beta:
                upper_bound = guess
            else:
                lower_bound = guess

        return guess, best_move

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
            best_value, best_move = self.iterative_deepening_mtd_fix(board)

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
