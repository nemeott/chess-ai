"""Chess game implementation with Pygame rendering and bots/players."""

import io
import math  # For quick render arrows
from typing import Literal

import cairosvg

# import chess
import chess.svg
import pygame
from bot5_rc import ChessBot
from constants_rc import (
    BLACK_USE_OPENING_BOOK,
    BREAK_TURN,
    CHECKING_MOVE_ARROW,
    IS_BOT,
    LAST_MOVE_ARROW,
    STARTING_FEN,
    UPDATE_DELAY_MS,
    WHITE_USE_OPENING_BOOK,
)
from PIL import Image
from score_rc import Score

import rust_chess as rc
from player import Player


class ChessGame:
    """Represents a chess game with Pygame rendering and player management."""

    __slots__ = [
        "WINDOW_SIZE",
        "arrow_move",
        "black_player",
        "board",
        "empty_board_surface",
        "highlighted_square_color",
        "last_board_state",
        "last_move",
        "last_update_time",
        "piece_images",
        "screen",
        "square_colors",
        "white_player",
    ]

    def __init__(self) -> None:
        """Initialize the ChessGame with board, players, and rendering setup."""
        if STARTING_FEN:
            self.board = rc.Board(STARTING_FEN)
        else:
            self.board = rc.Board()

        self.arrow_move: rc.Move | None = None  # Current move to draw an arrow for
        self.last_move: rc.Move | None = None  # Last move played
        self.last_update_time: int = pygame.time.get_ticks()

        # Initialize players based on IS_BOT flag
        if IS_BOT:
            self.white_player = ChessBot(WHITE_USE_OPENING_BOOK)
            self.black_player = ChessBot(BLACK_USE_OPENING_BOOK)
        else:
            self.white_player = Player(self, rc.WHITE)
            self.black_player = ChessBot(BLACK_USE_OPENING_BOOK)

        # Cache for piece images and board squares
        self.piece_images = {}

        # Use the default chess.svg colors (tan/brown)
        self.square_colors = {
            "light": pygame.Color("#f0d9b5"),  # Tan/cream squares (default chess.svg)
            "dark": pygame.Color("#b58863"),  # Brown squares (default chess.svg)
            "light_lastmove": pygame.Color("#cdd16a"),  # Highlighted light square for last move
            "dark_lastmove": pygame.Color("#aaa23b"),  # Highlighted dark square for last move
        }
        self.highlighted_square_color = pygame.Color(255, 255, 0, 128)  # Semi-transparent yellow

        # Initialize Pygame
        pygame.init()
        self.WINDOW_SIZE = 600
        self.screen = pygame.display.set_mode((self.WINDOW_SIZE, self.WINDOW_SIZE))
        pygame.display.set_caption("Chess Game")

        # Pre-render all piece images
        self.prerender_pieces()

        # Initialize last board state for optimized rendering
        self.last_board_state = None

        # Pre-render the empty board
        self.empty_board_surface = self.create_empty_board()

    def svg_to_pygame_surface(self, svg_string: str) -> pygame.Surface:
        """Convert SVG string to Pygame surface with optimized parameters."""
        # Reduce the resolution if it's just for pieces (they'll be scaled anyway)
        png_data = cairosvg.svg2png(
            bytestring=svg_string.encode("utf-8"),
            output_width=self.WINDOW_SIZE,  # Directly specify final size
            output_height=self.WINDOW_SIZE,
        )
        # Skip resizing step since we specified size in cairosvg
        if png_data is None:
            msg = "Failed to convert SVG to PNG."
            raise ValueError(msg)
        image = Image.open(io.BytesIO(png_data))
        mode = image.mode
        size = image.size
        data = image.tobytes()

        mode_literal: Literal["P", "RGB", "RGBX", "RGBA", "ARGB", "BGRA"] = mode  # type: ignore[assignment]
        return pygame.image.fromstring(data, size, mode_literal)

    def create_empty_board(self) -> pygame.Surface:
        """Create and cache the empty chess board with squares."""
        square_size = self.WINDOW_SIZE // 8
        surface = pygame.Surface((self.WINDOW_SIZE, self.WINDOW_SIZE))

        # Draw board squares
        for rank in range(8):
            for file in range(8):
                is_light = (file + rank) % 2 == 0
                square_color = self.square_colors["light" if is_light else "dark"]
                rect = pygame.Rect(file * square_size, rank * square_size, square_size, square_size)
                surface.fill(square_color, rect)

        return surface

    def prerender_pieces(self) -> None:
        """Pre-render all chess piece images at the correct size."""
        piece_symbols = ["p", "n", "b", "r", "q", "k", "P", "N", "B", "R", "Q", "K"]
        square_size = self.WINDOW_SIZE // 8

        # Create high resolution pieces then scale them once
        for symbol in piece_symbols:
            piece_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="500" height="500" viewBox="0 0 45 45">
                {chess.svg.PIECES[symbol]}
            </svg>"""

            # Convert to Pygame surface and pre-scale to square size
            piece_img = self.svg_to_pygame_surface(piece_svg)
            self.piece_images[symbol] = pygame.transform.scale(piece_img, (square_size, square_size))

    def fast_render_board(self, last_move: rc.Move | None = None, selected_square: rc.Square | None = None):
        """Render chess board using cached empty board and pieces."""
        board_state = self.board
        square_size = self.WINDOW_SIZE // 8

        # Start with a copy of the empty board
        surface = self.empty_board_surface.copy()

        # Only highlight last move squares and selected square
        if last_move:
            for square in [last_move.source, last_move.dest]:
                file = square.get_file()
                rank = 7 - square.get_rank()
                is_light = (file + rank) % 2 == 0
                square_color = self.square_colors["light_lastmove" if is_light else "dark_lastmove"]
                rect = pygame.Rect(file * square_size, rank * square_size, square_size, square_size)
                surface.fill(square_color, rect)

        # Highlight selected square
        if selected_square is not None:
            file = selected_square.get_file()
            rank = 7 - selected_square.get_rank()
            highlight_surf = pygame.Surface((square_size, square_size), pygame.SRCALPHA)
            highlight_surf.fill(self.highlighted_square_color)
            surface.blit(highlight_surf, (file * square_size, rank * square_size))

        # Draw pieces (now using pre-scaled pieces)
        for rank in range(8):
            for file in range(8):
                square = rc.Square.from_file_rank(file, 7 - rank)
                piece = board_state.get_piece_on(square)
                if piece:
                    piece = chess.Piece.from_symbol(piece.get_string())

                    # Directly use the pre-scaled piece image
                    surface.blit(self.piece_images[piece.symbol()], (file * square_size, rank * square_size))

        # Draw arrows if needed
        if LAST_MOVE_ARROW and last_move:
            # Using solid blue to match SVG arrow color
            self.draw_arrow(surface, last_move.source, last_move.dest, pygame.Color("#0000FF"))

        if CHECKING_MOVE_ARROW and self.arrow_move:
            # Using solid red to match SVG arrow color
            self.draw_arrow(surface, self.arrow_move.source, self.arrow_move.dest, pygame.Color("#FF0000"))

            return surface

        return None

    def draw_arrow(self, surface, from_square: rc.Square, to_square: rc.Square, color) -> None:
        """Draw an arrow that matches the SVG implementation."""
        square_size = self.WINDOW_SIZE // 8

        # Calculate start and end positions (centered in squares)
        from_file, from_rank = from_square.get_file(), 7 - from_square.get_rank()
        to_file, to_rank = to_square.get_file(), 7 - to_square.get_rank()

        # Match the SVG coordinate calculations
        xtail = (from_file + 0.5) * square_size
        ytail = (from_rank + 0.5) * square_size
        xhead = (to_file + 0.5) * square_size
        yhead = (to_rank + 0.5) * square_size

        # Calculate the direction vector
        dx, dy = xhead - xtail, yhead - ytail
        hypot = math.hypot(dx, dy)

        if hypot == 0:
            return  # Can't draw an arrow with zero length

        # Create semitransparent surface for entire arrow
        arrow_surface = pygame.Surface((self.WINDOW_SIZE, self.WINDOW_SIZE), pygame.SRCALPHA)

        # Use exact parameters from the SVG implementation
        marker_size = 0.75 * square_size
        marker_margin = 0.1 * square_size

        # Calculate shaft endpoint (where head begins)
        shaft_x = xhead - dx * (marker_size + marker_margin) / hypot
        shaft_y = yhead - dy * (marker_size + marker_margin) / hypot

        # Calculate arrowhead tip position (slightly inset from square edge)
        xtip = xhead - dx * marker_margin / hypot
        ytip = yhead - dy * marker_margin / hypot

        # Draw thick shaft - match SVG stroke width
        pygame.draw.line(
            arrow_surface,
            color,
            (xtail, ytail),
            (shaft_x, shaft_y),
            width=int(square_size * 0.2),  # Match the SVG stroke-width
        )

        # Calculate arrowhead points using SVG algorithm
        marker_points = [
            (xtip, ytip),  # Tip
            (shaft_x + dy * 0.5 * marker_size / hypot, shaft_y - dx * 0.5 * marker_size / hypot),
            (shaft_x - dy * 0.5 * marker_size / hypot, shaft_y + dx * 0.5 * marker_size / hypot),
        ]

        # Draw arrowhead
        pygame.draw.polygon(arrow_surface, color, marker_points)

        # Blit the arrow onto the main surface
        surface.blit(arrow_surface, (0, 0))

    def display_board(
        self,
        last_move: rc.Move | None = None,
        selected_square: rc.Square | None = None,
        force_update: bool = False,
    ) -> None:
        """Display the current board state with dynamic rendering selection."""
        current_time = pygame.time.get_ticks()
        # Skip update if too soon (unless forced)
        if (
            not force_update
            and hasattr(self, "last_update_time")
            and current_time - self.last_update_time < UPDATE_DELAY_MS
        ):
            return

        if CHECKING_MOVE_ARROW and self.arrow_move:
            # Use fast direct rendering during AI analysis
            board_surface = self.fast_render_board(last_move, selected_square)
            if board_surface is not None:
                self.screen.blit(board_surface, (0, 0))
            else:
                msg = "fast_render_board returned None, cannot blit to screen."
                raise ValueError(msg)
        else:
            # Use pretty SVG rendering during normal gameplay
            # Build highlight dictionary for the selected square
            highlight_squares = None
            if selected_square is not None:
                highlight_squares = {selected_square: {"fill": "#FFFF00", "stroke": "none"}}

            arrows = []
            if LAST_MOVE_ARROW and last_move:
                arrows.append(
                    chess.svg.Arrow(
                        # last_move.from_square,
                        # last_move.to_square,
                        last_move.source.get_index(),
                        last_move.dest.get_index(),
                        color="#0000FF",  # Blue color, solid
                    ),
                )
            if CHECKING_MOVE_ARROW and self.arrow_move:
                arrows.append(
                    chess.svg.Arrow(
                        # self.arrow_move.from_square,
                        # self.arrow_move.to_square,
                        self.arrow_move.source.get_index(),
                        self.arrow_move.dest.get_index(),
                        color="#FF0000",  # Red for checked move, solid
                    ),
                )

            # Create SVG with highlighted last move and selected square
            svg = chess.svg.board(
                board=chess.Board(fen=self.board.get_fen()),
                lastmove=chess.Move.from_uci(last_move.get_uci()) if last_move else None,
                squares=highlight_squares,
                arrows=arrows,
                size=self.WINDOW_SIZE,
                colors={
                    "square light": "#f0d9b5",  # Tan/cream
                    "square dark": "#b58863",  # Brown
                },
            )

            # Convert SVG to Pygame surface and display
            py_image = self.svg_to_pygame_surface(str(svg))
            self.screen.blit(py_image, (0, 0))

        pygame.display.flip()
        self.last_update_time = current_time

    def play_game(self) -> None:
        """Run the main game loop."""
        print("--------------------------------------------------------------")

        score = Score()
        score.initialize(self.board)  # Initialize scores once and update from there

        while self.board.get_status() == rc.BoardStatus.ONGOING:
            print(f"Player: {'White' if self.board.turn else 'Black'} - {self.board.fullmove_number}")

            # Determine current player
            current_player = self.white_player if self.board.turn else self.black_player

            # Display current board with highlights
            selected_square = getattr(current_player, "selected_square", None)
            self.display_board(self.last_move, selected_square, force_update=True)

            # Get actual score and update it for the current player
            score.initialize(self.board)
            current_player.set_score(score)

            # Get player's move
            move = current_player.get_move(self.board)

            if move is None:
                print("Game ended by player")
                break

            # Make the move
            self.board.reset_move_generator()
            if move in self.board.generate_legal_moves():
                self.board.make_move(move, check_legality=False)
            else:
                print(f"Illegal move attempted: {move}")
                break

            incremental_score = current_player.get_score()
            if incremental_score:  # If no incremental score, current player is human
                incremental_value = incremental_score.calculate()

                # Test if cached score is correct
                actual_score = Score()
                actual_score.initialize(self.board)
                actual_value = actual_score.calculate()

                print(f"Eval: {incremental_value}, {actual_value}")

                print(f"Move played: {move}")

                # Assert scores match
                assert incremental_score.material == actual_score.material, (
                    f"Material score mismatch: {incremental_score.material} != {actual_score.material}"
                )
                assert incremental_score.mg == actual_score.mg, (
                    f"Midgame score mismatch: {incremental_score.mg} != {actual_score.mg}"
                )
                assert incremental_score.eg == actual_score.eg, (
                    f"Endgame score mismatch: {incremental_score.eg} != {actual_score.eg}"
                )
                assert incremental_score.npm == actual_score.npm, (
                    f"Non-pawn material score mismatch: {incremental_score.npm} != {actual_score.npm}"
                )
                assert incremental_score.pawn_struct == actual_score.pawn_struct, (
                    f"Pawn structure score mismatch: {incremental_score.pawn_struct} != {actual_score.pawn_struct}"
                )
                assert incremental_score.king_safety == actual_score.king_safety, (
                    f"King safety score mismatch: {incremental_score.king_safety} != {actual_score.king_safety}"
                )

                assert incremental_value == actual_value, f"Score mismatch: {incremental_value} != {actual_value}"

            print("--------------------------------------------------------------")
            self.last_move = move

            if BREAK_TURN and self.board.fullmove_number > BREAK_TURN:
                pygame.quit()
                return

        # Display final position
        self.display_board(self.last_move, force_update=True)
        print(f"Number of turns: {self.board.fullmove_number}")  # Print number of turns
        result = self.board.get_status()
        print(f"Game Over! Result: {result}")

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
        pygame.quit()


if __name__ == "__main__":
    game = ChessGame()
    game.play_game()
