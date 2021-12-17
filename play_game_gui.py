import chess
import pygame as p
import argparse
import sys

import torch

from gui.gui_engine import GameState
from utils.util import move_to_coordinate, word_to_move

DIMENSION = 8
IMAGES = {}


def load_images(args):
    """Connect images to board state"""
    pieces = ['pawnw', 'knightw', 'bishopw', 'rookw', 'queenw', 'kingw',
              'pawnb', 'knightb', 'bishopb', 'rookb', 'queenb', 'kingb']
    for piece in pieces:
        IMAGES[piece] = p.transform.scale(p.image.load(f'gui/pieces/{piece}.png'), (args.width // 8, args.height // 8))


def draw_game_state(screen, gs, args):
    """Draws the game square"""
    draw_board(screen, args)  # Draw the black and white board
    draw_pieces(screen, args, gs.get_embedding_board())


def draw_board(screen, args, w_color='light gray', b_color='gray'):
    """Draws the chess board on screen"""
    colors = [p.Color(w_color), p.Color(b_color)]

    # Draw the board itself
    for row in range(DIMENSION):
        for column in range(DIMENSION):
            color = colors[(row + column) % 2]
            p.draw.rect(screen, color, p.Rect(column * args.width // 8, row * args.height // 8,
                                              args.width // 8, args.height // 8))


def draw_pieces(screen, args, embedding_board):
    """Drawing the pieces on the board"""

    embedding_board_flipped = torch.flip(embedding_board, [0])  #  For display

    for row in range(DIMENSION):
        for column in range(DIMENSION):

            # Check for pieces from embedding
            if embedding_board_flipped[row, column] in [7, 25]:
                piece = 'pawnw'  # white pawn
            elif embedding_board_flipped[row, column] in [8, 26]:
                piece = 'knightw'  # white knight
            elif embedding_board_flipped[row, column] in [9, 27]:
                piece = 'bishopw'  # white bishop
            elif embedding_board_flipped[row, column] in [10, 28]:
                piece = 'rookw'  # white rook
            elif embedding_board_flipped[row, column] in [11, 29]:
                piece = 'queenw'  # white queen
            elif embedding_board_flipped[row, column] in [12, 14, 30, 32]:
                piece = 'kingw'  # white king
            elif embedding_board_flipped[row, column] in [5, 23]:
                piece = 'pawnb'  # black pawn
            elif embedding_board_flipped[row, column] in [4, 22]:
                piece = 'knightb'  # black knight
            elif embedding_board_flipped[row, column] in [3, 21]:
                piece = 'bishopb'  # black bishop
            elif embedding_board_flipped[row, column] in [2, 20]:
                piece = 'rookb'  # black rook
            elif embedding_board_flipped[row, column] in [1, 19]:
                piece = 'queenb'  # black queen
            elif embedding_board_flipped[row, column] in [0, 15, 18, 33]:
                piece = 'kingb'  # black king
            else:
                piece = None

            if piece is not None:
                screen.blit(IMAGES[piece], p.Rect(column * args.width // 8, row * args.height // 8,
                                                  args.width // 8, args.height // 8))


def main(args):
    screen = p.display.set_mode((args.width, args.height))
    clock = p.time.Clock()
    screen.fill(p.Color('white'))

    square_size_x = args.width // 8
    square_size_y = args.height // 8

    # Access gamestate
    gs = GameState()
    load_images(args)

    # initialize variables
    sq_selected = ()
    player_clicks = []

    # game loop
    running = True
    while running:
        for e in p.event.get():

            keys = p.key.get_pressed()

            # Create the condition for exiting the loop
            if e.type == p.quit:
                running = False

            # ----------------------- KEYBOARD HANDLERS ----------------------------
            elif keys[p.K_z] and keys[p.K_LCTRL]:
                gs.undo_move()
                print('[INFO] Undoing move')
                sq_selected = ()
                player_clicks = []

            elif keys[p.K_SPACE]:
                print('[INFO] Opponent move')

            # ----------------------- MOUSE HANDLERS ----------------------------
            elif e.type == p.MOUSEBUTTONDOWN:
                mouse_location = p.mouse.get_pos()  # (x, y) of mouse location
                row, col = mouse_location[0] // square_size_x, mouse_location[1] // square_size_y

                # Check if I clicked on the same square twice
                if sq_selected == (row, col):
                    print('[INFO] cancelled')
                    sq_selected = ()
                    player_clicks = []
                else:
                    sq_selected = (row, col)
                    sq_displayed = (row + 1, 8 - col)
                    player_clicks.append(sq_selected)
                    print(f'[INFO] clicked: {sq_displayed}')

                # If there were 2 clicks for moves:
                if len(player_clicks) == 2:

                    valid_move = gs.make_move_mouse(player_clicks)
                    if not valid_move:
                        print('[WARN] Illegal move!!!')
                    sq_selected = ()  # Zero out the player inputs
                    player_clicks = []

        draw_game_state(screen, gs, args)
        clock.tick(args.max_fps)
        p.display.flip()

    p.quit()
    sys.exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--height', type=int, default=512, help='Screen height')
    parser.add_argument('--width', type=int, default=512, help='Screen width')
    parser.add_argument('--max_fps', type=int, default=60, help='Maximum frames-per-second')

    arg = parser.parse_args()
    main(arg)
