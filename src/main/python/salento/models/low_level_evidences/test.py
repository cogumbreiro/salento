#!/usr/bin/env python3
from salento.models.low_level_evidences.infer import *

import unittest
from unittest.mock import patch, Mock
class MockDist:
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
        self.calls.append(('infer_seq_iter', list(seq), resume is not None))
        for idx, (node, edge) in enumerate(seq):
            yield ARow(node=node, edge=edge, distribution=MockDist(), state=idx, cache_id=None)

class TestInfer(unittest.TestCase):

    def test_infer_states_ex(self):
        pred = BayesianPredictor(model=MockModel(), sess=None)
        pred._create_distribution = lambda x: x
        seq = [
            {'call': 'c1', 'states': ["c1_0", "c1_1"]},
            {'call': 'c2', 'states': []},
            {'call': 'c3', 'states': ["c3_0", "c3_1", "c3_2"]},
        ]
        for x in pred.infer_state_iter_ex('psi', seq):
            x
        self.assertEqual(pred.model.calls, [
            ('infer_seq_iter', [('START', 'V'), ('c1', 'H'), ('c2', 'H'), ('c3', 'H')], False),
            ('infer_seq_iter', [('c1', 'V'), ('0#c1_0', 'H'), ('1#c1_1', 'H')], True),
            ('infer_seq_iter', [('c2', 'V')], True),
            ('infer_seq_iter', [('c3', 'V'), ('0#c3_0', 'H'), ('1#c3_1', 'H'), ('2#c3_2', 'H')], True),
        ])


if __name__ == '__main__':
    unittest.main()

