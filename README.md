# Chess AI

A chess game implementation featuring bot vs Player (or another bot) gameplay. Contains many different search algorithms like MTD(f) safe fix, Negamax alpha beta, and Best Node Search (still WIP). Also contains other features like transposition tables, and move ordering heuristics. Built with Python, Pygame, and python-chess.

## Features

- **Player vs Bot Gameplay**: GUI with mouse controls, pawn promotion selection, and move highlighting.
- **Bot vs Bot Gameplay**: Have both bots compete against each other.
- **Advanced Search Algorithms**: MTD(f) variants, Negamax with alpha-beta pruning.
- **Transposition Tables**: LRU cache for storing evaluated positions.
- **Opening Book Support**: Polyglot format books for strong opening play.
- **Move Ordering Heuristics**: MVV/LVA capture ordering, promotions, and more in the works.
- **Game State Detection**: Checkmate, stalemate, threefold repetition (WIP), insufficient material, fifty-move rule.
- **Configurable Settings**: Adjustable search depth, table sizes, piece values, and more.
- **Debugging Tools**: Move arrows, search statistics.
- **Bot Comparison Suite**: WIP testing on Bratko-Kopec positions for performance evaluation.

## Usage

### Playing the Game

1. Clone the repository:

```sh
git clone https://github.com/nemeott/chess-ai.git
cd chess-ai
```

2. Set up the virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Or

uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

3. Run the program:

```sh
python game.py
```

#### Human vs Bot Mode

To play as a human against the bot:

1. Set `IS_BOT = False` in `constants.py`
2. Run the game
3. Click on pieces to select and move them
4. The AI will respond with its move
5. For pawn promotions, a selection menu will appear

#### Bot vs Bot Mode (Default)

To watch two AIs play against each other:

1. Set `IS_BOT = True` in `constants.py`
2. Run the game
3. The game will run automatically, displaying moves and board updates
4. Useful for testing and demonstration purposes

## Project Structure

- `game.py`: Main game loop, board rendering, and user interface.
- `player.py`: Human player class handling input and moves.
- `bot5.py`: Bot with MTD(f) safe fix, transposition tables, and other optimizations.
- `score.py`: Position evaluation and scoring logic.
- `tt_entry.py`: Transposition table entry definitions.
- `constants.py`: Game constants like search depth, piece values.
- `colors.py`: ANSI color codes for console output.
- `requirements.txt`: Python dependencies.

## Configuration

Key constants can be adjusted in `constants.py`:

### Search Settings

- `DEPTH`: Search depth for the bot (default: 5)
- `TT_SIZE`: Transposition table size in MB (default: 32)
- `OPENING_BOOK_PATH`: Path to Polyglot opening book file (optional)
- `WHITE_USE_OPENING_BOOK` / `BLACK_USE_OPENING_BOOK`: Enable/disable opening book for each color

### Game Settings

- `IS_BOT`: Set to `True` for bot vs bot mode, `False` for human vs bot
- `STARTING_FEN`: Custom starting position (FEN string, default: standard position)
- `LAST_MOVE_ARROW`: Display arrow highlighting the last move

### Debug Settings

- `CHECKING_MOVE_ARROW`: Display arrows for moves being checked (debug mode)
- `UPDATE_DELAY_MS`: Delay between visual updates in milliseconds (default: 30)
  - Only really required when when using the move arrow since displaying every move checked is intensive
- `RENDER_DEPTH`: Depth to render checking moves (set to `DEPTH` for root moves)
- `BREAK_TURN`: Break after a specific number of turns (for debugging)

## Evaluation Function

The position evaluation function assesses the relative strength of a chess position using several components:

- **Material Balance**: Piece values derived from Stockfish (pawn: 208, knight: 781, bishop: 825, rook: 1276, queen: 2538, king: 32000 centipawns)
- **Piece-Square Tables**: Position-dependent bonuses/penalties for each piece type, with separate tables for midgame and endgame phases (tables from Rofchade)
- **Pawn Structure**: Penalties for isolated pawns (-20) and doubled pawns (-10)
- **Bishop Pair Bonus**: Additional value (half a pawn) for having both bishops on the board
- **Tapered Evaluation**: Smooth interpolation between midgame and endgame scores based on remaining non-pawn material

## Technical Details

- **Data Structures**: Numpy arrays for piece-square tables and numerical computations; LRU cache for transposition tables; doubly-linked lists (llist) for move history and repetition detection
- **Performance Optimizations**: Function caching, incremental score updates during search, move ordering heuristics (MVV/LVA, killer moves), null window searches in MTD(f)
- **Memory Management**: Configurable transposition table sizes, efficient board state copying using python-chess
- **Rendering**: SVG-based piece rendering converted to Pygame surfaces using CairoSVG and PIL
- **Testing Framework**: Automated bot comparison on Bratko-Kopec test positions using multiprocessing for parallel evaluation

## License

MIT License
