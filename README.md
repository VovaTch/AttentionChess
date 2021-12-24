# AttentionChess
A personal project for creating a transformer encoder-decoder-based chess engine. This project is based on PyTorch and Python Chess. I use an Encoder-Decoder architecture by encoding the chess board as words with positional encoding. The queries are all the legal moves (coded as 4864 possible moves in UCI format), and the output is a logit probability vector for all the moves.

[Weights available in Google Drive](https://drive.google.com/file/d/1JnyL1bIrFSKIEePJ6xFfT3gP9rARwT-q/view?usp=sharing)

## Current playing instructions:

* Run `python play_game_gui.py` for a simple chess board gui.
* Move pieces by selecting them and pressing the destination square.
* Press twice on the same square to cancel the move.
* Press **space** for the bot to perform a move.
* **ctrl+z** to undo.
* When promoting, a display of promotion options will appear, press on the wanted one.
* Move the king 2 squares to the right or left to castle.
* At the end of the game, the program will crash, **WIP** of course.
