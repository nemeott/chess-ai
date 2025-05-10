class OpeningBook:
    def init(self, book_path=None):
        self.book = {}  # Dictionary: FEN -> list of (move, weight) tuples
        if book_path:
            self.load_book(book_path)

    def load_book(self, book_path):
        """Load opening book from file"""
        # Polyglot book loading implementation
        try:
            with open(book_path, "rb") as book_file:
                while True:
                    entry_data = book_file.read(16)
                    if len(entry_data) < 16:
                        break

                    key = int.from_bytes(entry_data[0:8], byteorder="big")
                    move = int.from_bytes(entry_data[8:10], byteorder="big")
                    weight = int.from_bytes(entry_data[10:12], byteorder="big")
                    learn = int.from_bytes(entry_data[12:16], byteorder="big")

                    # Convert key to FEN, move to UCI notation
                    fen = self._key_to_fen(key)  # Simplified, actual implementation is complex
                    uci_move = self._bin_to_uci(move)

                    if fen in self.book:
                        self.book[fen].append((uci_move, weight))
                    else:
                        self.book[fen] = [(uci_move, weight)]
        except Exception as e:
            print(f"Error loading opening book: {e}")

    def get_move(self, board, variation=0.1):
        """Get a move from the opening book for the current position"""
        try:
            # Get simplified FEN (only piece positions and turn)
            fen = self._simplified_fen(board)

            if fen in self.book:
                moves = self.book[fen]

                # Sort by weight, highest first
                moves.sort(key=lambda x: x[1], reverse=True)

                # Introduce variation if requested
                if variation > 0 and len(moves) > 1:
                    # Sometimes pick not the best move but a random good one
                    if random.random() < variation:
                        # Pick from top 3 moves with probability proportional to weight
                        top_moves = moves[:min(3, len(moves))]
                        total_weight = sum(weight for , weight in topmoves)
                        r = random.random() * total_weight
                        cumulative = 0
                        for move, weight in top_moves:
                            cumulative += weight
                            if r <= cumulative:
                                return move

                # Default: return highest weighted move
                return moves[0][0]

            return None  # No book move available
        except Exception as e:
            print(f"Error getting book move: {e}")
            return None

    def simplifiedfen(self, board):
        """Return simplified FEN (position and turn only)"""
        full_fen = board.fen()
        return ' '.join(full_fen.split(' ')[:2])
