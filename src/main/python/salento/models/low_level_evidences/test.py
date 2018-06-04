#!/usr/bin/env python3

import unittest
import time

################################################################################
# data_reader.py
from salento.models.low_level_evidences.data_reader import *

def get_seq_paths_old(js, idx=0):
    # The original algorithm for `get_seq_paths`:
    if idx == len(js):
       return [[('STOP', SIBLING_EDGE)]]
    call = js[idx]['call']
    pv = [[(call, CHILD_EDGE)] +
          [('{}#{}'.format(i, state), SIBLING_EDGE) for i, state in enumerate(js[idx]['states'])] +
          [('STOP', SIBLING_EDGE)]]
    ph = [[(call, SIBLING_EDGE)] + path for path in get_seq_paths_old(js, idx + 1)]
    return pv + ph

class DataReaderExamples(unittest.TestCase):
    js0 = []

    js1 = [
        {'call': 'call1', 'states': [1,2,3]},
        {'call': 'call2', 'states': [2,3,4,11,22,33]},
        {'call': 'call3', 'states': [5,6,7,8,9]},
    ]

    js2 = [
        {'call': 'call1', 'states': [1,2,3]},
        {'call': 'call2', 'states': [2,3,4,11,22,33]},
        {'call': 'call3', 'states': [5,6,7,8,9]},
        {'call': 'call1', 'states': [1,2,3]},
        {'call': 'call2', 'states': [2,3,4,11,22,33]},
        {'call': 'call3', 'states': [5,6,7,8,9]},
        {'call': 'call1', 'states': []},
        {'call': 'call2', 'states': [2,3,4,11,22,33]},
        {'call': 'call3', 'states': [5,6,7,8,9]},
        {'call': 'call1', 'states': [1,2,3]},
        {'call': 'call2', 'states': [2,3,4,11,22,33]},
        {'call': 'call3', 'states': [5,6,7,8,9]},
        {'call': 'call1', 'states': [1,2,3]},
        {'call': 'call2', 'states': [2,3,4,11,22,33]},
        {'call': 'call3', 'states': [5,6,7,8,9]},
    ]

    def run_test(self, js, debug=False):
        start = time.time()
        res1 = get_seq_paths_old(js)
        end = time.time()
        if debug: print("get_seq_path_old", end - start)

        start = time.time()
        res2 = get_seq_paths(js)
        end = time.time()
        if debug: print("get_seq_path", end - start)

        self.assertEqual(res1, res2)

    def test_example1(self):
        self.run_test(self.js1)

    def test_example2(self):
        self.run_test(self.js2)

    def test_example0(self):
        self.run_test(self.js0)

    def test_benchmark(self):
        self.run_test(self.js1 * 300, debug=True)


################################################################################
# infer.py
from salento.models.low_level_evidences.infer import *

class MockDist:
    def __init__(self, *args):
        self.args = args

    def __getitem__(self, key):
        return 0.5

class Vocabs:
    def __getitem__(self, key):
        return 0

class Decoder:
    def __init__(self):
        self.vocab = Vocabs()

class Config:
    def __init__(self):
        self.decoder = Decoder()

ARow = namedtuple('Row', ['node', 'edge', 'distribution', 'state', 'cache_id'])

class MockModel:
    def __init__(self):
        self.calls = []
        self.config = Config()

    def infer_seq(self, sess, psi, seq, cache=None, resume=None):
        seq = list(seq)
        dist = {}
        path = ""
        for step in self.infer_seq_iter(sess, psi, seq, cache, resume):
            dist = step.distribution
            path = step.cache_id
        return dist

    def infer_seq_iter(self, sess, psi, seq, cache=None, resume=None):
        seq = list(seq)
        self.calls.append((seq, resume is not None))
        for idx, (node, edge) in enumerate(seq):
            dist = MockDist(seq[:idx + 1])
            yield ARow(node=node, edge=edge, distribution=dist, state=idx, cache_id=None)

class TestInfer(unittest.TestCase):

    def test_infer_states_ex(self):
        pred = BayesianPredictor(model=MockModel(), sess=None)
        pred._create_distribution = lambda x: x
        seq = [
            {'call': 'c1', 'states': ["c1_0", "c1_1"]},
            {'call': 'c2', 'states': []},
            {'call': 'c3', 'states': ["c3_0", "c3_1", "c3_2"]},
        ]
        result = list(pred.infer_state_iter('psi', seq))
        self.assertEqual(len(seq) + 1, len(result))
        # We convert the result into strings just so we can test it more easily
        # Each row pairs the term name with the sequence that yielded the given
        # distribution.
        dists = list(list((n,d.args[0]) for (n,d) in x) for x in result)
        self.assertEqual(
            [
                ('c1', [('START', 'V')]),
                ('0#c1_0', [('c1', 'V')]), # state 0
                ('1#c1_1', [('c1', 'V'), ('0#c1_0', 'H')]), # state 1
                # The sentinel None marks the end of the states
                (None, [('c1', 'V'), ('0#c1_0', 'H'), ('1#c1_1', 'H')]), # end of states
            ],
            dists[0]
        )

        self.assertEqual(
            [
                ('c2', [('START', 'V'), ('c1', 'H')]),
                (None, [('c2', 'V')]), # End of states
            ],
            dists[1]
        )

        self.assertEqual(
            [
                ('c3', [('START', 'V'), ('c1', 'H'), ('c2', 'H')]),
                ('0#c3_0', [('c3', 'V')]),
                ('1#c3_1', [('c3', 'V'), ('0#c3_0', 'H')]),
                ('2#c3_2', [('c3', 'V'), ('0#c3_0', 'H'), ('1#c3_1', 'H')]),
                (None, [('c3', 'V'), ('0#c3_0', 'H'), ('1#c3_1', 'H'), ('2#c3_2', 'H')]),
            ],
            dists[2]
        )
        # The result has one more element than the input as we are also checking
        # for the end of sequence token. By default, the sentinel token is None.
        # Notice how the sentinel term has no state-information.
        self.assertEqual(
            [
                (None, [('START', 'V'), ('c1', 'H'), ('c2', 'H'), ('c3', 'H')])
            ],
            dists[3]
        )

        # We ensure that the side-effects are sound:
        self.assertEqual(pred.model.calls, [
            # The distribution of each call
            ([('START', 'V'), ('c1', 'H'), ('c2', 'H'), ('c3', 'H')], False),
            # The distribution of each state for the 0-th call
            ([('c1', 'V'), ('0#c1_0', 'H'), ('1#c1_1', 'H')], True),
            # The distribution of each state for the 1-st call
            ([('c2', 'V')], True),
            # The distribution of each state for the 2-nd call
            ([('c3', 'V'), ('0#c3_0', 'H'), ('1#c3_1', 'H'), ('2#c3_2', 'H')], True),
        ])


if __name__ == '__main__':
    unittest.main()

