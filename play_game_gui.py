import chess
import pygame as p
import argparse
import sys
import collections

import torch

import model.attchess as module_arch
from gui.gui_engine import GameState
from utils.util import prepare_device, board_to_embedding_coord
from parse_config import ConfigParser

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


def draw_promotion_choice(screen, args, color='dark gray'):
    p.draw.rect(screen, color, p.Rect(2.5 * args.width // 8, 2.5 * args.height // 8,
                                      args.width // 8, args.height // 8))
    screen.blit(IMAGES['queenw'], p.Rect(2.5 * args.width // 8, 2.5 * args.height // 8,
                                         args.width // 8, args.height // 8))

    p.draw.rect(screen, color, p.Rect(4.5 * args.width // 8, 4.5 * args.height // 8,
                                      args.width // 8, args.height // 8))
    screen.blit(IMAGES['rookw'], p.Rect(4.5 * args.width // 8, 4.5 * args.height // 8,
                                         args.width // 8, args.height // 8))

    p.draw.rect(screen, color, p.Rect(2.5 * args.width // 8, 4.5 * args.height // 8,
                                      args.width // 8, args.height // 8))
    screen.blit(IMAGES['bishopw'], p.Rect(2.5 * args.width // 8, 4.5 * args.height // 8,
                                         args.width // 8, args.height // 8))

    p.draw.rect(screen, color, p.Rect(4.5 * args.width // 8, 2.5 * args.height // 8,
                                      args.width // 8, args.height // 8))
    screen.blit(IMAGES['knightw'], p.Rect(4.5 * args.width // 8, 2.5 * args.height // 8,
                                         args.width // 8, args.height // 8))


def promotion_selector(args):
    selected = None

    while selected is None:
        mouse_location = p.mouse.get_pos()

        # Select the queen
        if 2.5 * args.width // 8 <= mouse_location[0] <= 3.5 * args.width // 8 and \
                2.5 * args.width // 8 <= mouse_location[1] <= 3.5 * args.width // 8:
            selected = 'q'

        # Select the rook
        if 4.5 * args.width // 8 <= mouse_location[0] <= 5.5 * args.width // 8 and \
                4.5 * args.width // 8 <= mouse_location[1] <= 5.5 * args.width // 8:
            selected = 'r'

        # Select the bishop
        if 2.5 * args.width // 8 <= mouse_location[0] <= 3.5 * args.width // 8 and \
                4.5 * args.width // 8 <= mouse_location[1] <= 5.5 * args.width // 8:
            selected = 'b'

        # Select the knight
        if 4.5 * args.width // 8 <= mouse_location[0] <= 5.5 * args.width // 8 and \
                2.5 * args.width // 8 <= mouse_location[1] <= 3.5 * args.width // 8:
            selected = 'n'

    return selected


def draw_pieces(screen, args, embedding_board):
    """Drawing the pieces on the board"""

    embedding_board_flipped = torch.flip(embedding_board, [0])  # For display

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


def main(args, config):

    # Load the net
    logger = config.get_logger('Chess game')
    device, _ = prepare_device(config['n_gpu'])

    # Load network
    engine = config.init_obj('arch', module_arch)
    checkpoint = torch.load(config.resume, map_location=torch.device('cpu'))
    engine.load_state_dict(checkpoint['state_dict'])
    engine = engine.to(device).eval()
    logger.info('Engine loaded')

    # Prepare the screen of the gui
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
    promotion_flag = False
    selected_piece = None  # Selected piece for promotion

    # game loop
    running = True
    while running:
        for e in p.event.get():

            keys = p.key.get_pressed()

            # Create the condition for exiting the loop
            if e.type == p.quit:
                running = False

            # ----------------------- KEYBOARD HANDLERS ----------------------------
            elif e.type == p.KEYDOWN:

                if keys[p.K_z] and keys[p.K_LCTRL]:
                    gs.undo_move()
                    print('[INFO] Undoing move')
                    sq_selected = ()
                    player_clicks = []

                elif keys[p.K_SPACE]:
                    print('[INFO] Bot move')

                    # Run the network and get a move sample
                    board_tensor = board_to_embedding_coord(gs.board).to(device)
                    outputs_legal, outputs_class_vec = engine(board_tensor.unsqueeze(0))
                    legal_move_list, cls_vec, endgame_flag = engine.post_process(outputs_legal, outputs_class_vec)
                    cat = torch.distributions.Categorical(cls_vec[0])

                    # Force legal move
                    while True:

                        sample_idx = cat.sample()
                        sample = legal_move_list[0][sample_idx]

                        print(f'[INFO] Move in uci: {sample}')
                        if sample in gs.board.legal_moves:
                            gs.board.push(sample)
                            break

            # ----------------------- MOUSE HANDLERS ----------------------------
            elif e.type == p.MOUSEBUTTONDOWN:

                if promotion_flag:
                    selected_piece = promotion_selector(args)
                    print(selected_piece)

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

                if promotion_flag:
                    player_clicks.pop()
                # If there were 2 clicks for moves:
                if len(player_clicks) == 2:

                    if not promotion_flag:
                        promotion_flag = gs.check_promotion(player_clicks)
                    else:
                        promotion_flag = False

                    valid_move = gs.make_move_mouse(player_clicks, promotion=selected_piece)
                    selected_piece = None
                    if not valid_move:
                        print('[WARN] Illegal move!!!')
                    sq_selected = ()  # Zero out the player inputs
                    if not promotion_flag:
                        player_clicks = []

        draw_game_state(screen, gs, args)
        if promotion_flag:
            draw_promotion_choice(screen, args)
        clock.tick(args.max_fps)
        p.display.flip()

    p.quit()
    sys.exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple chess board for playing the bot.')
    parser.add_argument('-c', '--config', default='config_s1.json', type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default='checkpoint-epoch190.pth', type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default='cuda', type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('--height', type=int, default=512, help='Screen height')
    parser.add_argument('--width', type=int, default=512, help='Screen width')
    parser.add_argument('--max_fps', type=int, default=60, help='Maximum frames-per-second')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]

    config = ConfigParser.from_args(parser, options)

    arg = parser.parse_args()
    main(arg, config)
