import chess

from score import Score
import bot3
import bot4

from typing import Optional

fens = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" # Starting position
]

bot3_wins = []
bot4_wins = []
draws = []

for fen in fens:
    for turn in [True, False]:
        board = chess.Board(fen)

        score = Score()
        score.initialize(board)

        white = bot3.ChessBot() if turn else bot4.ChessBot()
        black = bot4.ChessBot() if turn else bot3.ChessBot()

        while not board.is_game_over():
            print("---------------------------------------------------")

            white.set_score(score)

            move = white.get_move(board)
            print(f"Move played: {move}")
            board.push(move)

            score = white.get_score()
            print(f"Eval: {score.calculate()}")

            print("---------------------------------------------------")

            score.initialize(board)
            black.set_score(score)

            move = black.get_move(board)
            print(f"Move played: {move}")
            board.push(move)

            score = black.get_score()
            print(f"Eval: {score.calculate()}")

        print("---------------------------------------------------")

        print(f"Number of turns: {board.fullmove_number}") # Print number of turns
        outcome: Optional[chess.Outcome] = board.outcome()
        print(f"Game Over! Result: {outcome}")

        if outcome:
            match outcome.winner:
                case chess.WHITE: # White wins
                    if turn: # Bot 3 won
                        bot3_wins.append((outcome.termination, board.fullmove_number))
                    else: # Bot 4 won
                        bot4_wins.append((outcome.termination, board.fullmove_number))
                case chess.BLACK: # Black wins
                    if turn: # Bot 4 won
                        bot4_wins.append((outcome.termination, board.fullmove_number))
                    else: # Bot 3 won
                        bot3_wins.append((outcome.termination, board.fullmove_number))
                case None:
                    draws.append((outcome.termination, board.fullmove_number))
        else:
            print("No outcome returned.")

print("---------------------------------------------------")

print(f"Bot 3 won {len(bot3_wins)} times.")
print(f"Bot 4 won {len(bot4_wins)} times.")
print(f"Draws: {len(draws)}")

print("---------------------------------------------------")

print(f"Bot 3 wins: {bot3_wins}")
print(f"Bot 4 wins: {bot4_wins}")
print(f"Draws: {draws}")
