import random


PIECE_LIST = ["R", "N", "B", "Q", "P"]

 
def _init_board():
	board = [[" " for x in range(8)] for y in range(8)]
	return board

 
def _place_kings(brd):
	while True:
		rank_white, file_white, rank_black, file_black = random.randint(0,7), random.randint(0,7), random.randint(0,7), random.randint(0,7)
		diff_list = [abs(rank_white - rank_black),  abs(file_white - file_black)]
		if sum(diff_list) > 2 or set(diff_list) == set([0, 2]):
			brd[rank_white][file_white], brd[rank_black][file_black] = "K", "k"
			break
 
 
def _populate_board(brd, wp, bp):
	for x in range(2):
		if x == 0:
			piece_amount = wp
			pieces = PIECE_LIST
		else:
			piece_amount = bp
			pieces = [s.lower() for s in PIECE_LIST]
		while piece_amount != 0:
			piece_rank, piece_file = random.randint(0, 7), random.randint(0, 7)
			piece = random.choice(pieces)
			if brd[piece_rank][piece_file] == " " and _pawn_on_promotion_square(piece, piece_rank) == False:
				brd[piece_rank][piece_file] = piece
				piece_amount -= 1
    
    
def _fen_from_board(brd):
	fen = ""
	for x in brd:
		n = 0
		for y in x:
			if y == " ":
				n += 1
			else:
				if n != 0:
					fen += str(n)
				fen += y
				n = 0
		if n != 0:
			fen += str(n)
		fen += "/" if fen.count("/") < 7 else ""
	fen += " w - - 0 1\n"
	return fen
 
 
def _pawn_on_promotion_square(pc, pr):
	if pc == "P" and pr == 0:
		return True
	elif pc == "p" and pr == 7:
		return True
	return False


def generate_position():
	piece_amount_white, piece_amount_black = random.randint(0, 15), random.randint(0, 15)
	board = _init_board()
	_place_kings(board)
	_populate_board(board, piece_amount_white, piece_amount_black)
	return _fen_from_board(board)