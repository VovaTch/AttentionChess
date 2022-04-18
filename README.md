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

* Blow up the parameters until what my 2060 can handle, with an option to external training on colab.
* Make the Jump-Start-RL style learning work, maybe in a need for debugging and hyperparameter tuning.
* Fighting against the impossible. It seem making a specific playstyle is a stretch as AlphaZero was trained on an insane hardware and it would take me thousands of years to even make this competitive if I were to train it like DeepMind.\
* Make a docker/colab such that anyone can play the game.
