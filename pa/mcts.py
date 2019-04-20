import random

try:
    import tabulate
    tabulate.PRESERVE_WHITESPACE = True
    from tabulate import tabulate as tab
    TABULATE = True
except ImportError:
    TABULATE = False


PLAYOUT = 'PLAYOUT'
ANALIZE = 'ANALIZE'
WAIT = 'WAIT'
PLAYOUT_EXCEEDED = 'PLAYOUT_EXCEEDED'


def make_result(value):
    if value is None:
        return None

    if isinstance(value, list):
        return [None] + value
    else:
        return [None, value, -value]


def dump_nodes(nodes):
    if TABULATE:
        rows = (node.dump_row() for node in nodes)
        columns = ['view', 'locked', 'score', 'qgames', 'eval', 'qmoves', 'es.value', 'es.weight' ] + [ 'r' + str(i) for i in range(1, 10) ]
        return tab(rows, tablefmt='plain', headers=columns)
    else:
        return '\n'.join('\t'.join(str(value) for value in node.dump_row()) for node in nodes)

def print_nodes(nodes):
    print(dump_nodes(nodes))


class Job:
    def __init__(self, jid, action, node=None, state=None, moves=[]):
        self.jid = jid
        self.action = action
        self.node = node
        self.state = state
        self.moves = moves

    def dump_node_view(self):
        return self.node.view if self.node else ''

    def dump_state(self):
        state = self.state
        if state is None:
            return ''
        return state.dump() if hasattr(state, 'dump') else str(state)

    def dump_moves(self, *, sep=' '):
        return sep.join(self.moves) if self.moves else ''

    def dump_row(self, *, sep=' '):
        return [
            self.jid,
            self.action,
            self.dump_node_view(),
            self.dump_state(),
            self.dump_moves(),
        ]

    def dump(self):
        return ' '.join(str(item) for item in self.dump_row())


class Estimation:
    def __init__(self, value=0, weight=0):
        self.value = value
        self.weight = weight

    def dump_value(self):
        return '{:5.2f}'.format(self.value) if self.weight != 0 else '   - '

    def dump_weight(self):
        return '{:4.2f}'.format(self.weight) if self.weight != 0 else '  - '

    def dump_row(self):
        return [ self.dump_value(), self.dump_weight() ]

    def dump(self):
        return ':'.join(self.dump_row())


class Node:
    def __init__(self, view):
        self.view = view
        self.locked = False
        self.score = 0
        self.qgames = 0
        self.estimation = Estimation()
        self.result = None
        self.moves = []
        self.children = []

    def lock(self):
        assert not self.locked
        self.locked = True

    def unlock(self):
        assert self.locked
        self.locked = False

    def dump_locked(self):
        return 'LOCKED' if self.locked else ''

    def dump_evaluation(self):
        return '{:5.2f}'.format(self.score / self.qgames) if self.qgames > 0 else '  -  '

    def dump_moves(self):
        return str(len(self.moves))

    def dump_result(self):
        if self.result is None:
            return []
        return [ str(value) for value in self.result[1:] ]

    def dump_row(self):
        return [
            self.view, self.dump_locked(), self.score, self.qgames, self.dump_evaluation(),
            self.dump_moves(), *self.estimation.dump_row(), *self.dump_result()
        ]

    def dump(self):
        return ' '.join(str(item) for item in self.dump_row())


class RandomMoveSelection:
    def __call__(self, nodes):
        return random.choice(nodes)

class MctsServer:
    def __init__(self, root_state, *, select_best=RandomMoveSelection()):
        self.root_state = root_state.copy()
        self.nodes = {}
        self.jobs = {}
        self.current_jid = 0
        self.select_best = select_best

    def get_node(self, view):
        node = self.nodes.get(view)
        if node is None:
            node = Node(view)
            self.nodes[view] = node
        return node

    def new_job(self, action, node=None, state=None, moves=[]):
        if action not in [WAIT, PLAYOUT_EXCEEDED]:
            self.current_jid += 1
            jid = self.current_jid
        else:
            jid = 0

        if node:
            node.lock()

        job = Job(jid, action, node, state, moves)
        if jid > 0:
            self.jobs[jid] = job
        return job

    def apply_playout(self, moves, result):
        state = self.root_state.copy()

        view = state.get_view(state.active)
        node = self.get_node(view)
        node.qgames += 1

        for move in moves:
            view = state.view_move(move)
            node = self.get_node(view)
            node.qgames += 1
            node.score += result[state.active]
            state.do_move(move)

        node.result = result[:]

    def get_job(self, *, max_playouts=99):
        qplayouts = 0
        while True:
            job = self.try_get_job()
            if job.action != PLAYOUT_EXCEEDED:
                return job
            qplayouts += 1
            if max_playouts is not None and qplayouts > max_playouts:
                return job

    def try_get_job(self):
        state = self.root_state.copy()
        moves = []
        while True:
            active = state.active
            view = state.get_view(active)
            node = self.get_node(view)

            if node.result:
                self.apply_playout(moves, node.result)
                return self.new_job(PLAYOUT_EXCEEDED)

            if node.locked:
                return self.new_job(WAIT)

            if node.qgames == 0:
                return self.new_job(PLAYOUT, node, state, moves)

            if not node.moves:
                return self.new_job(ANALIZE, node, state)

            choices = [ state.view_move(move) for move in node.moves ]
            nodes = [ self.get_node(view) for view in choices ]
            alternatives = [ node for node in nodes if not node.locked ]
            if not alternatives:
                return self.new_job(WAIT)

            best_node = self.select_best(alternatives)
            best_index = nodes.index(best_node)
            best_move = node.moves[best_index]

            moves.append(best_move)
            state.do_move(best_move)

    def put_job(self, jid, data):
        job = self.jobs.get(jid)
        if job is None:
            raise ValueError('Bad job ID = {} - not found.'.format(jid))
        if job.action != data['action']:
            raise ValueError('Bad job action {} for job with ID = {}: expected {}'.format(data['action'], jid, job.action))

        if job.action == ANALIZE:
            self.put_analize(job, data)
        elif job.action == PLAYOUT:
            self.put_playout(job, data)
        else:
            raise ValueError('Bad action {} for job'.format(job.action))

        del self.jobs[jid]
        if job.node:
            job.node.unlock()

    def put_playout(self, job, data):
        result = make_result(data.get('result'))
        moves = job.moves + data.get('moves', [])
        self.apply_playout(moves, result)

    def put_analize(self, job, data):
        result = make_result(data.get('result'))
        if result:
            job.node.result = result
            return

        moves = [ str(move) for move in data.get('moves', []) ]
        estimations = data.get('estimations', [])

        assert moves
        assert len(moves) == len(estimations)

        job.node.moves = moves

        for move, estimation in zip(moves, estimations):
            view = job.state.view_move(move)
            child = self.get_node(view)
            child.estimation = Estimation(*estimation)
            job.node.children.append(child)

    def dump_nodes(self, nodes=None):
        return dump_nodes(nodes or self.nodes.values())

    def dump_view(self, view):
        node = self.nodes.get(view)
        if node is None:
            return '<Not found view {}>'.format(view)
        return dump_nodes([node] + node.children)

    def dump_jobs(self):
        return '\n'.join(job.dump() for job in self.jobs.values())

    def print_nodes(self):
        print(self.dump_nodes())

    def print_view(self, view):
        print(self.dump_view(view))

    def print_jobs(self):
        print(self.dump_jobs())
