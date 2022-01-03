import chess
import pygame as p
import argparse
import sys
import collections
import platform
import pathlib

import torch

import model.attchess as module_arch
from gui.gui_engine import GameState
from utils.util import prepare_device, board_to_embedding_coord, move_to_coordinate
from parse_config import ConfigParser

DIMENSION = 8
IMAGES = {}


def load_images(args):
    """Connect images to board state"""
    pieces = ['pawnw', 'knightw', 'bishopw', 'rookw', 'queenw', 'kingw',
              'pawnb', 'knightb', 'bishopb', 'rookb', 'queenb', 'kingb']
    for piece in pieces:
        IMAGES[piece] = p.transform.scale(p.image.load(f'gui/pieces/{piece}.png'), (args.width // 8, args.height // 8))


def draw_game_state(screen, gs, args, player_clicks):
    """Draws the game square"""
    draw_board(screen, args)  # Draw the black and white board
    highlight_sqares(screen, gs, args, player_clicks)
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


def draw_text(screen, args, text):
    """Writes the text on the screen"""

    # init
    p.font.init()

    # Black border
    font = p.font.SysFont("urwgothic", 32, True, False)
    text_object = font.render(text, 0, p.Color('black'))
    text_location = p.Rect(0, 0, args.width, args.height).move(args.width / 2 - text_object.get_width() / 2,
                                                               args.height / 2 - text_object.get_height())
    screen.blit(text_object, text_location.move(2, 2))
    screen.blit(text_object, text_location.move(2, -2))
    screen.blit(text_object, text_location.move(-2, 2))
    screen.blit(text_object, text_location.move(-2, -2))

    # White innards
    font = p.font.SysFont("urwgothic", 32, True, False)
    text_object = font.render(text, 0, p.Color('light yellow'))
    text_location = p.Rect(0, 0, args.width, args.height).move(args.width / 2 - text_object.get_width() / 2,
                                                               args.height / 2 - text_object.get_height())
    screen.blit(text_object, text_location)


def highlight_sqares(screen, gs:GameState, args, player_clicks,
                     highlight_piece_color='red', highlight_move_color='orange'):
    """Draws highlights for pieces and possible moves"""
    if len(player_clicks) == 0:
        return

    move_coor_from = player_clicks[0][0] + 8 * (7 - player_clicks[0][1])

    # Collect all the moves to be highlighted
    display_list = []
    for move in gs.board.legal_moves:
        if move.from_square == move_coor_from:
            display_list.append(move)

    # If no square was selected
    if len(display_list) == 0:
        return

    # If square was selected
    surface = p.Surface((args.width // 8, args.height // 8))
    surface.set_alpha(100)
    surface.fill(p.Color(highlight_piece_color))

    screen.blit(surface, (player_clicks[0][0] * args.width // 8, player_clicks[0][1] * args.height // 8))

    surface.fill(p.Color(highlight_move_color))
    for move in display_list:
        move_word = move.to_square
        column, row = int(move_word % 8), int(move_word // 8)
        screen.blit(surface, (column * args.width // 8, (7 - row) * args.height // 8))


def animate_move(screen, gs: GameState, args, move: chess.Move, clock, frames_per_square=3,
                 w_color='light gray', b_color='gray'):

    colors = [p.Color(w_color), p.Color(b_color)]

    # Convert move to coordinates
    from_square = move.from_square
    to_square = move.to_square
    row_from, colomn_from = int(from_square % 8), int(from_square // 8)
    row_to, column_to = int(to_square % 8), int(to_square // 8)

    # Creating the mid coordinates
    d_r = row_to - row_from
    d_c = column_to - colomn_from
    frame_count = (abs(d_r) + abs(d_c)) * frames_per_square
    for frame in range(frame_count + 1):
        row_mid, column_mid = row_from + d_r * frame / frame_count, colomn_from + d_c * frame / frame_count

        draw_board(screen, args, w_color=w_color, b_color=b_color)
        embedded_board = board_to_embedding_coord(gs.board)
        draw_pieces(screen, args, embedded_board)

        # Erase the moved piece
        color = colors[(row_to + (1 + column_to)) % 2]
        p.draw.rect(screen, color, p.Rect(row_to * args.width // 8, (7 - column_to) * args.height // 8,
                                          args.width // 8, args.height // 8))

        # Draw the moving piece
        embedded_board_flipped = torch.flip(embedded_board, [0])
        piece = extract_piece_from_embedding(embedded_board[column_to, row_to])
        screen.blit(IMAGES[piece], p.Rect(row_mid * args.width // 8, (7 - column_mid) * args.height // 8,
                                          args.width // 8, args.height // 8))

        # Display on screen
        p.display.flip()
        clock.tick(args.max_fps)


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

            piece = extract_piece_from_embedding(embedding_board_flipped[row, column])

            if piece is not None:
                screen.blit(IMAGES[piece], p.Rect(column * args.width // 8, row * args.height // 8,
                                                  args.width // 8, args.height // 8))


def extract_piece_from_embedding(emb):
    # Check for pieces from embedding
    if emb in [7, 25]:
        piece = 'pawnw'  # white pawn
    elif emb in [8, 26]:
        piece = 'knightw'  # white knight
    elif emb in [9, 27]:
        piece = 'bishopw'  # white bishop
    elif emb in [10, 28]:
        piece = 'rookw'  # white rook
    elif emb in [11, 29]:
        piece = 'queenw'  # white queen
    elif emb in [12, 14, 30, 32]:
        piece = 'kingw'  # white king
    elif emb in [5, 23]:
        piece = 'pawnb'  # black pawn
    elif emb in [4, 22]:
        piece = 'knightb'  # black knight
    elif emb in [3, 21]:
        piece = 'bishopb'  # black bishop
    elif emb in [2, 20]:
        piece = 'rookb'  # black rook
    elif emb in [1, 19]:
        piece = 'queenb'  # black queen
    elif emb in [0, 15, 18, 33]:
        piece = 'kingb'  # black king
    else:
        piece = None

    return piece


def main(args, config):

    # Windows boiler code stuff
    if platform.system() == 'Windows':
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath

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
    undo_flag = False

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
                    undo_flag = True

                elif keys[p.K_SPACE]:
                    print('[INFO] Bot move')

                    # Run the network and get a move sample
                    outputs_legal, outputs_class_vec = engine.board_forward([gs.board])
                    legal_move_list, cls_vec, endgame_flag = engine.post_process(outputs_legal, outputs_class_vec)
                    legal_move_san = {gs.board.san(legal_move): float(cls_prob) for legal_move, cls_prob
                                      in zip(legal_move_list[0], cls_vec[0])}
                    print(f'[INFO] Probabilities for moves: {legal_move_san}')
                    cat = torch.distributions.Categorical(cls_vec[0])

                    print(f'[INFO] Endgame flag: {endgame_flag[0]:.2g}')
                    # Force legal move
                    while True:

                        sample_idx = cat.sample()
                        # sample_idx = torch.argmax(cls_vec[0])
                        sample = legal_move_list[0][sample_idx]

                        print(f'[INFO] Move in uci: {sample}')
                        if sample in gs.board.legal_moves:
                            gs.board.push(sample)
                            break
                        
                elif keys[p.K_r] and keys[p.K_LCTRL]:
                    gs.reset()
                    sq_selected = ()
                    player_clicks = []
                    promotion_flag = False
                    selected_piece = None  # Selected piece for promotion
                    undo_flag = False

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
                    else:
                        animate_move(screen, gs, args, gs.board.peek(), clock)
                        undo_flag = False
                    sq_selected = ()  # Zero out the player inputs
                    if not promotion_flag:
                        player_clicks = []

        draw_game_state(screen, gs, args, player_clicks)
        if not undo_flag:
            if gs.is_white_checkmate:
                draw_text(screen, args, 'WHITE CHECKMATE')
            elif gs.is_black_checkmate:
                draw_text(screen, args, 'BLACK CHECKMATE')
            elif gs.is_draw:
                draw_text(screen, args, 'DRAW')
            undo_flag = False

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
    parser.add_argument('-r', '--resume', default='model_best_init.pth', type=str,
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
