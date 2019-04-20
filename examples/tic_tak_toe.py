from pa.mcts import MctsServer
from itertools import repeat


LINES = [ (1, 2, 3), (4, 5, 6), (7, 8, 9), (1, 4, 7), (2, 5, 8), (3, 6, 9), (1, 5, 9), (3, 5, 7) ]

class TikTakToe:
    def __init__(self):
        self.cells = dict(zip(range(1,10), repeat('.')))
        self.active = 1

    def copy(self):
        instance = TikTakToe()
        instance.cells = self.cells.copy()
        instance.active = self.active
        return instance

    def active_symbol(self):
        return 'X' if self.active == 1 else 'O'

    def format_view(self, getter):
        row_fmt = ''.join(['{}'] * 3)
        fmt = '|'.join([row_fmt] * 3)
        args = [ getter(i) for i in range(1,10) ]
        return fmt.format(*args)

    def get_view(self, player=None):
        return self.format_view(lambda cell: self.cells[cell])

    def do_move(self, move):
        move = int(move)
        self.cells[move] = self.active_symbol()
        self.active ^= 3

    def view_move(self, move):
        move = int(move)
        getter = lambda cell: self.cells[cell] if cell != move else self.active_symbol()
        return self.format_view(getter)

    def gen_moves(self):
        return [ str(pair[0]) for pair in self.cells.items() if pair[1] == '.' ]

    def get_result(self):
        for line in LINES:
            view = [ self.cells[index] for index in line ]
            if view == [ 'X', 'X', 'X' ]:
                return 1
            if view == [ 'O', 'O', 'O' ] :
                return -1
        free = [ ch for ch in self.cells.values() if ch == '.' ]
        return 0 if len(free) == 0 else None

    def dump(self):
        return self.get_view()
