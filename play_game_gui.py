import chess
from numpy import argmax
import pygame as p
from pygame import gfxdraw
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
from model.score_functions import ScoreWinFast
from data_loaders.game_roll import InferenceMoveSearcher, InferenceBoardNode
()
DIMENSION = 8
IMAGES = {}


def load_images(args):
    """Connect images to board state"""
    pieces = ['pawnw', 'knightw', 'bishopw', 'rookw', 'queenw', 'kingw',
              'pawnb', 'knightb', 'bishopb', 'rookb', 'queenb', 'kingb']
    if args.use_vanilla_pieces:
        for piece in pieces:
            IMAGES[piece] = p.transform.smoothscale(p.image.load(f'gui/pieces/{piece}.png'), (args.width // 8, args.height // 8))
    else:
        for piece in pieces:
            IMAGES[piece] = p.transform.smoothscale(p.image.load(f'gui/pieces/{piece}d.png'), (args.width // 8, args.height // 8))
            
    IMAGES['background'] = p.transform.smoothscale(p.image.load(f'gui/pieces/Desert_board.png'), (args.width, args.height))


def draw_game_state(screen, gs, args, player_clicks, flip_board):
    """Draws the game square"""
    draw_board(screen, args)  # Draw the black and white board
    highlight_sqares(screen, gs, args, player_clicks, flip_board)
    draw_pieces(screen, args, gs.get_embedding_board(), flip_board)


def draw_rect_alpha(surface, color, rect, alpha):
    "Draws a rectangle with transparency"
    shape_surf = p.Surface(p.Rect(rect).size)
    shape_surf.set_alpha(alpha * 256)
    p.draw.rect(shape_surf, color, shape_surf.get_rect())
    surface.blit(shape_surf, rect)


def draw_board(screen, args, w_color='light gray', b_color='gray', alpha=0.35):
    """Draws the chess board on screen"""
    colors = [p.Color(w_color), p.Color(b_color)]

    # Draw the board itself
    screen.blit(IMAGES['background'], p.Rect(0, 0, args.width, args.height))
    for row in range(DIMENSION):
        for column in range(DIMENSION):
            color = colors[(row + column) % 2]
            alpha_multiplier = 1.5 if (row + column) % 2 == 0 else 1.0
            draw_rect_alpha(screen, color, (column * args.width // 8, row * args.height // 8,
                                              args.width // 8, args.height // 8), alpha=alpha * alpha_multiplier)

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


def highlight_sqares(screen, gs:GameState, args, player_clicks_origin, flip_board,
                     highlight_piece_color='red', highlight_move_color='orange'):
    """Draws highlights for pieces and possible moves"""
    if len(player_clicks_origin) == 0:
        return
    
    if flip_board:
        player_clicks = [[7 - player_clicks_origin[0][0], 7 - player_clicks_origin[0][1]]]
    else:
        player_clicks = player_clicks_origin
    
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

    screen.blit(surface, (player_clicks_origin[0][0] * args.width // 8, player_clicks_origin[0][1] * args.height // 8))

    surface.fill(p.Color(highlight_move_color))
    for move in display_list:
        move_word = move.to_square
        column, row = int(move_word % 8), int(move_word // 8)
        if flip_board:
            column, row = 7 - column, 7 - row
        screen.blit(surface, (column * args.width // 8, (7 - row) * args.height // 8))


def animate_move(screen, gs: GameState, args, move: chess.Move, clock, flip_board, frames_per_square=3,
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
        
        if flip_board:
            row_mid, column_mid = 7 - row_mid, 7 - column_mid

        draw_board(screen, args, w_color=w_color, b_color=b_color)
        embedded_board = board_to_embedding_coord(gs.board)
        draw_pieces(screen, args, embedded_board, flip_board)

        # Erase the moved piece
        color = colors[(row_to + (1 + column_to)) % 2]
        if not flip_board:
            p.draw.rect(screen, color, p.Rect(row_to * args.width // 8, (7 - column_to) * args.height // 8,
                                            args.width // 8, args.height // 8))
        else:
            p.draw.rect(screen, color, p.Rect((7 - row_to) * args.width // 8, (column_to) * args.height // 8,
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


def draw_pieces(screen, args, embedding_board, flip_board):
    """Drawing the pieces on the board"""

    if flip_board:
        embedding_board_flipped = torch.flip(embedding_board, [1])  # For display
    else:
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


@torch.no_grad()
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
    move_searcher = InferenceMoveSearcher(engine=engine, num_pruned_moves=3)

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
    score_function = ScoreWinFast(100)
    flip_board = False

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

                # Undo move
                if keys[p.K_z] and keys[p.K_LCTRL]: 
                    gs.undo_move()
                    print('[INFO] Undoing move')
                    sq_selected = ()
                    player_clicks = []
                    undo_flag = True

                # Perform computer move
                elif keys[p.K_SPACE]:
                    print('[INFO] Bot move')

                    # Run the network and get a move sample
                    outputs_legal, outputs_class_vec, value = engine([gs.board])
                    legal_move_list, cls_vec, value_full = engine.post_process(outputs_legal, outputs_class_vec, 
                                                                               value, num_pruned_moves=None)
                    legal_move_san = {gs.board.san(legal_move): float(cls_prob) for legal_move, cls_prob
                                      in zip(legal_move_list[0], cls_vec[0])}
                    print(f'[INFO] Probabilities for moves: {legal_move_san}')
                    
                    # Initiate node for move searcher
                    move_node = InferenceBoardNode(gs.board, None, score_function, cls_vec[0], value_full[0], device=device)
                    move_node.legal_move_list = legal_move_list[0]

                    value_np = value_full.detach().numpy()[0]
                    print(f'[INFO] Board value: {value_np}')

                    sample = move_searcher(move_node, args.leaves, min_depth=args.min_depth)
                    move_searcher.reset()
                    # sample = legal_move_list[0][torch.argmax(cls_vec[0]).int()]

                    print(f'[INFO] Move in uci: {sample}')
                    gs.make_move_uci(sample)
                    animate_move(screen, gs, args, gs.board.peek(), clock, flip_board)
                    undo_flag = False
                    
                # Reset board    
                elif keys[p.K_r] and keys[p.K_LCTRL]:
                    gs.reset()
                    sq_selected = ()
                    player_clicks = []
                    promotion_flag = False
                    selected_piece = None  # Selected piece for promotion
                    undo_flag = False
                    
                # Flip board
                elif keys[p.K_f] and keys[p.K_LCTRL]:
                    flip_board = not flip_board
                    
                # Quit game
                elif keys[p.K_q] and keys[p.K_LCTRL]:
                    running = False

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
                        promotion_flag = gs.check_promotion(player_clicks, flip_board)
                    else:
                        promotion_flag = False

                    valid_move = gs.make_move_mouse(player_clicks, promotion=selected_piece, flip_board=flip_board)
                    selected_piece = None
                    if not valid_move:
                        print('[WARN] Illegal move!!!')
                    else:
                        animate_move(screen, gs, args, gs.board.peek(), clock, flip_board=flip_board)
                        undo_flag = False
                    sq_selected = ()  # Zero out the player inputs
                    if not promotion_flag:
                        player_clicks = []

        draw_game_state(screen, gs, args, player_clicks, flip_board=flip_board)
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
    parser.add_argument('-l', '--leaves', default=10, type=int, help='Number of leaf nodes for move search')
    parser.add_argument('--min_depth', type=int, default=5, help='Minimum computation depth')
    parser.add_argument('--height', type=int, default=1000, help='Screen height')
    parser.add_argument('--width', type=int, default=1000, help='Screen width')
    parser.add_argument('--max_fps', type=int, default=60, help='Maximum frames-per-second')
    parser.add_argument('--exp_prob', type=float, default=0.25, help='Move search exploration probability ' +
                        'for deviating from the arg-max policy')
    parser.add_argument('--use_vanilla_pieces', action='store_true', 
                        help='Flag for using default pieces instead of self painted ones')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]

    config = ConfigParser.from_args(parser, options)

    arg = parser.parse_args()
    main(arg, config)
