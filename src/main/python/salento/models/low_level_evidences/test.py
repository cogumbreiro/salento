#!/usr/bin/env python3

# Copyright 2018 Georgia Tech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import time

################################################################################
# data_reader.py
from salento.models.low_level_evidences.data_reader import *
from typing import *

def get_seq_paths_old(js, idx=0):
    # Copyright 2017 Rice University
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
    js0 : List[Dict[str, Any]] = []

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
        old_time = end - start

        start = time.time()
        res2 = get_seq_paths(js)
        end = time.time()
        new_time = end - start

        self.assertEqual(res1, res2)
        if debug: print("get_seq_path speedup:", "{:.2f}x".format(old_time/new_time))

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
from salento.models.low_level_evidences import model

class DistMock:
    def __init__(self, path):
        # We record the arguments given upon creation
        # This is currently used to record the path of the distribution
        self.path = path

    def __getitem__(self, key):
        # The probability return is not really used
        return 0.5

class VocabsMock:
    # Returns an identifier given a vocab; the return value does not really
    # matter
    def __getitem__(self, key):
        return 0

class DecoderMock:
    # The `BayesianPredictor` only accesses the vocabs.
    def __init__(self):
        self.vocab = VocabsMock()

class ConfigMock:
    # A config object with the minimal capabilities of the config obj
    # used by `BayesianPredictor`.
    def __init__(self):
        self.decoder = DecoderMock()

class MockModel:
    def __init__(self):
        # The calls are used to capture the side-effects of handling a Model.
        self.calls = []
        # The config object is accessed directly by `BayesianPredictor`,
        # hence the need to mock it
        self.config = ConfigMock()

    # Use the original `Model.infer_seq` method:
    infer_seq = Model.infer_seq

    def infer_seq_iter(self, sess, psi, seq, cache=None, resume=None):
        # We fake the invocation of the model `infer_seq_iter` and just return
        # an collection of `Row` objects.
        seq = list(seq)
        # Log which calls were executed
        self.calls.append((seq, resume is not None))
        for idx, (node, edge) in enumerate(seq):
            # In each distribution, we record the path used to compute this
            # distribution
            dist = DistMock(seq[:idx + 1])
            yield model.Row(node=node, edge=edge, distribution=dist, state=idx, cache_id=None)

class TestInfer(unittest.TestCase):

    def test_infer_states(self):
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
        dists = list(list((n,d.path) for (n,d) in x) for x in result)
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

