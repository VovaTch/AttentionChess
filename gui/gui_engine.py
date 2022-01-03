import torch
import numpy
import chess
import pygame as p

from utils.util import board_to_embedding_coord, word_to_move


class GameState:
    """Game state based on tutorial from Eddie Sharik; as I use python-chess,
    probably it will just be some sort of a wrapper."""
    def __init__(self):
        self.board = chess.Board()
        self.move_log = []
        self.cap_white = []
        self.cap_black = []
        self.is_white_checkmate = False
        self.is_black_checkmate = False
        self.is_draw = False
        
    def reset(self):
        self.board = chess.Board()
        self.move_log = []
        self.cap_black = []
        self.cap_black = []
        self.is_white_checkmate = False
        self.is_black_checkmate = False
        self.is_draw = False

    def get_embedding_board(self):
        """Gets the embedding of the base board"""
        return board_to_embedding_coord(self.board)

    def make_move(self, move_uci: str):
        move_made = chess.Move.from_uci(move_uci)
        self.board.push(move_made)
        self.move_log.append(move_made)

    def check_promotion(self, player_clicks):
        """Checks for promotions"""
        pieces_moved_enc = self.get_embedding_board()[(7 - player_clicks[0][1]), player_clicks[0][0]]

        if (pieces_moved_enc in [7, 25] and player_clicks[1][1] == 0)\
                or (pieces_moved_enc in [5, 23] and player_clicks[1][1] == 7):
            return True
        else:
            return False

    def make_move_mouse(self, player_clicks, promotion=None):
        """Makes a chess move from user input"""
        move_coor_from = player_clicks[0][0] + 8 * (7 - player_clicks[0][1])
        pieces_moved_enc = self.get_embedding_board()[(7 - player_clicks[0][1]), player_clicks[0][0]]
        move_coor_to = player_clicks[1][0] + 8 * (7 - player_clicks[1][1])
        pieces_captured_enc = self.get_embedding_board()[(7 - player_clicks[1][1]), player_clicks[1][0]]

        # Pawn promotions
        if player_clicks[1][1] in [0, 7] and promotion is not None:
            diff = player_clicks[0][0] - player_clicks[1][0]
            # Queen
            if promotion == 'q':
                move_coor_to = 65 - diff
            # Rook
            if promotion == 'r':
                move_coor_to = 68 - diff
            # Bishop
            if promotion == 'b':
                move_coor_to = 71 - diff
            # Knight
            if promotion == 'n':
                move_coor_to = 74 - diff

        move_word = move_coor_from + 64 * move_coor_to
        move_chess = word_to_move(int(move_word + 1e-6))
        if move_chess in self.board.legal_moves:
            self.board.push(move_chess)
        else:
            return False

        if not self.board.turn:  # Capture logging
            if self.convert_embedding_to_piece(pieces_captured_enc) != '-':
                self.cap_white.append(self.convert_embedding_to_piece(pieces_captured_enc))
        else:
            if self.convert_embedding_to_piece(pieces_captured_enc) != '-':
                self.cap_black.append(self.convert_embedding_to_piece(pieces_captured_enc))

        # Endgame conditions
        if self.board.is_checkmate():
            if not self.board.turn:
                self.is_white_checkmate = True
            else:
                self.is_black_checkmate = True
        elif self.board.is_stalemate() or self.board.is_repetition() or self.board.is_seventyfive_moves():
            self.is_draw = True
        else:
            self.is_white_checkmate = False
            self.is_black_checkmate = False
            self.is_draw = False

        return True

    def undo_move(self):
        """Undo move"""
        try:
            self.board.pop()
        except:
            print('[WARN] Trying to undo initial position')


    @staticmethod
    def convert_embedding_to_piece(embedding_square):

        # Check for pieces from embedding
        if embedding_square in [7, 16, 25, 34]:
            piece = 'pawnw'  # white pawn
        elif embedding_square in [8, 26]:
            piece = 'knightw'  # white knight
        elif embedding_square in [9, 27]:
            piece = 'bishopw'  # white bishop
        elif embedding_square in [10, 28]:
            piece = 'rookw'  # white rook
        elif embedding_square in [11, 29]:
            piece = 'queenw'  # white queen
        elif embedding_square in [12, 14, 30, 32]:
            piece = 'kingw'  # white king
        elif embedding_square in [5, 17, 23, 35]:
            piece = 'pawnb'  # black pawn
        elif embedding_square in [4, 22]:
            piece = 'knightb'  # black knight
        elif embedding_square in [3, 21]:
            piece = 'bishopb'  # black bishop
        elif embedding_square in [2, 20]:
            piece = 'rookb'  # black rook
        elif embedding_square in [1, 19]:
            piece = 'queenb'  # black queen
        elif embedding_square in [0, 15, 18, 33]:
            piece = 'kingb'  # black king
        else:
            piece = '-'  # Empty square

        return piece
