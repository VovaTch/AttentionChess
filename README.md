# AttentionChess
A personal project for creating a transformer encoder-decoder-based chess engine. This project is based on PyTorch and Python Chess. I use an Encoder-Decoder architecture by encoding the chess board as words with positional encoding. The queries are all the legal moves (coded as 4864 possible moves in UCI format), and the output is a logit probability vector for all the moves.
