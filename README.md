# AttentionChess

<img src="https://github.com/VovaTch/AttentionChess/blob/main/Attchess.png" alt="drawing" width="500"/>


A personal project for creating a transformer encoder-decoder-based chess engine. This project is based on PyTorch and Python Chess. I use an Encoder-Decoder architecture by encoding the chess board as words with positional encoding. The queries are all the legal moves (coded as 4864 possible moves in UCI format), and the output is a logit probability vector for all the moves, and a value from -100 to 100 that determines the state of the board the closer to 100, white is better, the closer to -100 black is better. A drawing possition is 0.

[Weights available in Google Drive](https://drive.google.com/file/d/1JnyL1bIrFSKIEePJ6xFfT3gP9rARwT-q/view?usp=sharing)

## Current playing instructions:

* Run `python play_game_gui.py` for a simple chess board gui.
* Move pieces by selecting them and pressing the destination square. A highlight for the piece and the legal moves will appear.
* Press twice on the same square to cancel the move.
* When promoting, a display of promotion options will appear, press on the wanted piece.
* Move the king 2 squares to the right or left to castle.

### Hotkeys:

* Press **space** for the bot to perform a move.
* **ctrl+z** to undo.
* **ctrl+f** to flip board.
* **ctrl+r** to reset game.
* **ctrl+q** to quit the game.

## Current WIP: 

* Anchor-based learning scheme. Learn endgame positions, then generate random positions and learn from the leaves of the search tree, finally train fully self play from the initial board position.
* Incorporate FFCV for faster training.
* Clean the code bloat, make RippleLinear publically available if used (not likely).
