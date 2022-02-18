import chess

class ScoreWinFast:
    """
    Basic score to win as fast as possible
    """
    def __init__(self, moves_to_end, score_max=100):
        self.last_move_idx = moves_to_end
        self.score_max = score_max

    def __call__(self, move_idx, *args):
        score = self.score_max / ((self.last_move_idx - move_idx + 1) // 2)
        return score
    
class ScoreScaling:
    """
    Scaling score like a standard RL reward function
    """
    def __init__(self, moves_to_end, score_max=100) -> None:
        self.last_move_idx = moves_to_end
        self.score_max = score_max
        
    def __call__(self, move_idx, *args):
        score = self.score_max * 0.95 ** (self.last_move_idx - move_idx - 1)
        return score