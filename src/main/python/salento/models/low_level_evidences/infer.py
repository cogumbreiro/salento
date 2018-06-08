# Copyright 2018 Georgia Tech
# Copyright 2017-2018 Rice University
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

"""
The infer module introduces `BayesianPredictor` an abstraction layer over
`low_level_evidences.model.Model`, customized for inference. The main
class introduced is `BayesianPredictor`, which is built using an instance
of a `Model` and a TensorFlow session. The main role of this module is to
translate from a sequence of terms into the internal representation that
is consumed by TensorFlow.
"""

from __future__ import print_function
from typing import *

import tensorflow as tf
import numpy as np

import os
import json

from salento.models.low_level_evidences import model

from salento.models.low_level_evidences.model import Model
from salento.models.low_level_evidences.utils import CHILD_EDGE, SIBLING_EDGE
from salento.models.low_level_evidences.utils import read_config

from collections import namedtuple

# Types used in the class
# A term is a string
Term = str
# The probability distribution of each term
TermDist = Dict[Term, float]
# The cache used in the model
Cache = Dict[str, Any]
# A JSON call-event
Event = Dict[str, Any]


try:
    class Entry(NamedTuple):
        """
        Pairs a term with a term distribution probability.
        Can be used a `Tuple[Term, TermDist]`.
        """
        term: Term
        distribution: TermDist

except NameError:
    # Python < 3.6
    Entry = namedtuple('Entry', ['term', 'distribution'])

def event_states(call:Event) -> Iterable[Term]:
    for i, elem in enumerate(call['states']):
        key = '{}#{}'.format(i, elem)
        yield key

def _next_state(event:Event) -> Iterable[Tuple[str,str]]:
    yield (event['call'], CHILD_EDGE)
    for key in event_states(event):
        yield (key, SIBLING_EDGE)

def _next_call(event:Event) -> Tuple[str, str]:
    return (event['call'], SIBLING_EDGE)

def _sequence_to_graph(sequence:List[Event], step='call') -> List[Tuple[str,str]]:
    seq = [('START', CHILD_EDGE)]
    seq.extend(_next_call(call) for call in sequence[:-1])
    if len(sequence) > 0:
        if step == 'call':
            seq.append(_next_call(sequence[-1]))
        elif step == 'state':
            seq.extend(_next_state(sequence[-1]))
        else:
            raise ValueError('invalid step: {}'.format(step))
    return seq

def _call_names(terms:List[Event], sentinel:Term) -> Iterable[Term]:
    yield from (x['call'] for x in terms)
    yield sentinel

T = TypeVar('T')

class VectorMapping(Dict[T,float]):
    """
    A `VectorMapping` exposes a vector of values as a map from terms to values.

    Given:
    1. a list of terms (that acts as a map from integers (ids) to terms),
    2. a mapping from terms to identifiers (what's the identifier of this term?), and
    3. a vector from identifier to value (eg, a list of probabilities)

    A `VectorMapping` acts as a map from terms to values (eg, probabilities).
    """
    def __init__(self, data:List[float], id_to_term:List[T], term_to_id:Dict[T,int]) -> None:
        self.data = data
        self.id_to_term = id_to_term
        self.term_to_id = term_to_id

    def keys(self):
        return self.term_to_id.keys()

    def __iter__(self):
        return iter(self.keys())

    def values(self):
        return self.data

    def __contains__(self, key):
        return key in self.term_to_id

    def get(self, key, default=None):
        if key in self:
            return self[key]
        else:
            return default

    def items(self):
        return zip(self.id_to_term, self.data)

    def __getitem__(self, key):
        return self.data[self.term_to_id[key]]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return repr(dict(self.items()))

def iter_append(elems:Iterable[T], elem:T) -> Iterable[T]:
    yield from elems
    yield elem

class BayesianPredictor:
    """
    The `BayesianPredictor` offers two main capabilities, both of which
    take a call-sequence as an input and output an iterator to navigate the
    distribution probabilities of the given call-sequence.

    Method `Model.infer_call_iter` offers the simplest capability: it allows the
    user to iterate over the distribution probability given a sequence of calls and
    **ignores** the states of each term. The return value of `Model.infer_call_iter`
    is an iterator of pairs that contain the term name (a string) and the
    probability distribution.

    For instance, given an instance `pred` of `BayesianPredictor`, we can yield
    the probability of each term in in a sequence of calls `calls` with the
    following code:

    ```python
    for (call, dist) in pred.distribution_call_iter(spec, calls, sentinel=self.END_MARKER):
        yield dist[call]
    ```

    Method `Model.infer_state_iter` allows the user to iterate over the
    distribution probability given a sequence of calls, **including** the states
    of each term. The return value of `Model.infer_state_iter`
    is an iterator of lists; each list pairs the term name (a string) with the
    probability distribution. The first element of each list is the call name
    and the distribution of the call name, the subsequent pairs consist of the
    distribution probability for each state of that call name.

    For instance, say that we want to "flatten" the output of
    `Model.infer_state_iter`. In the following code we yield the probability
    of each call name and of each state in a single iterator.

    ```python
    for row in pred.distribution_state_iter(spec, calls, sentinel=self.END_MARKER):
        for (term, dist) in row:
            yield dist[term]
    ```
    """
    @classmethod
    def load(cls:Type['BayesianPredictor'], save:str, sess:tf.Session):
        """
        Takes a path `save` where it locates the configuration file
        `config.json` and then uses such a file load a `Model` for inference.
        Afterwhich invokes the `restore` method.
        """
        # load the saved config
        with open(os.path.join(save, 'config.json')) as f:
            config = read_config(json.load(f), chars_vocab=True)
        pred = cls(sess=sess, model=Model(config, True))
        pred.restore(save)
        return pred

    def __init__(self, model:Model, sess:tf.Session) -> None:
        """
        `model` is an instance of `Model`.
        `sess` is a TensorFlow session.
        """
        self.model = model
        self.sess = sess

    def restore(self, save:str):
        """
        Uses the current session to restore from the checkpoint given
        in directory `save`.
        """
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(save)
        saver.restore(self.sess, ckpt.model_checkpoint_path)

    # step can be 'call' or 'state', depending on if you are looking for distribution over the next call/state
    def infer_step(self, psi, sequence:List[Event], step='call', cache=None):
        seq = _sequence_to_graph(sequence, step)
        dist = self.model.infer_seq(self.sess, psi, seq, cache=cache)
        return self._create_distribution(dist)

    def _create_distribution(self, dist:List[float]) -> TermDist:
        return VectorMapping(dist, self.model.config.decoder.chars, self.model.config.decoder.vocab)

    def psi_random(self):
        return np.random.normal(size=[1, self.model.config.latent_size])

    def psi_from_evidence(self, js_evidences):
        return self.model.infer_psi(self.sess, js_evidences)

    def infer_call_iter(self, psi, sequence:List[Event], sentinel:Term, cache:Cache=None) \
            -> Iterable[Entry]:
        """
        Yields a sequence that pairs each call with the term probability
        distribution at that position. The return is a sequence of entries,
        each of which pairs a call name with the distribution probability,
        suffixed by the sentinel token.

        That is, given a list of 2 events:

            [{'call': 'foo', 'states': ['S']}, {'call': 'bar': 'states': []}]

        The return will be a sequence of 3 entries:

            ('foo', {...})
            ('bar', {...})
            (sentinel, {...})

        where each `{...}` is a distinct distribution probability.
        """
        sequence = list(sequence) # cache the terms
        seq = _sequence_to_graph(sequence=sequence, step='call')
        r_call_names = _call_names(sequence, sentinel)
        r_dist = (self._create_distribution(row.distribution) \
            for row in self.model.infer_seq_iter(self.sess, psi, seq, cache=cache))
        return (Entry(n,d) for (n,d) in zip(r_call_names, r_dist))


    def infer_state_iter(self, psi, sequence:List[Event], sentinel:Term, cache:Cache=None) \
            -> Iterable[List[Entry]]:
        """
        The length of output sequence has one more element than the input
        sequence (the sentinel).

        Each list of entries contains the distribution probability of the call
        followed by each state, and is terminated with a sentinel in the case
        where there is state. In particular, the sentinel call has no state
        information, so it consists of one entry alone.
        Additionally, each state is encoded with the following format: the
        state position, the state separator `#`, and the state data.

        That is given a list of 2 events:

            [{'call': 'foo', 'states': ['S']}, {'call': 'bar': 'states': []}]

        the output will be a generator with three lists:

            [('foo', {...}), ('0#S', {...}), (sentinel, {...})]
            [('bar', {...}), (sentinel, {...})]
            [(sentinel, {...})]

        where each `{...}` is a distinct term distribution.

        Since call `foo` has only one state at position 0 with data `S`,
        therefore the encoded state becomes `0#S`.
        """
        sequence = list(sequence)
        seq = _sequence_to_graph(sequence=sequence, step='call')

        states = None

        r_call_names = _call_names(sequence, sentinel)
        r_dist = []
        r_states = []

        for idx, row in enumerate(self.model.infer_seq_iter(self.sess, psi, seq, cache=cache)):
            r_dist.append(self._create_distribution(row.distribution))

            new_states = []

            if idx < len(sequence):
                call = sequence[idx]
                new_seq = list(_next_state(call))
                dists = self.model.infer_seq_iter(self.sess, psi, new_seq, cache=cache, resume=row)
                for (key, entry) in zip(iter_append(event_states(call), None), dists): #type: Tuple[Optional[str], model.Row]
                    dist = self._create_distribution(entry.distribution)
                    if key is None:
                        new_states.append(Entry(sentinel, dist))
                    else:
                        new_states.append(Entry(key, dist))


            r_states.append(new_states)

        for name, dist, states in zip(r_call_names, r_dist, r_states):
            line = [Entry(name, dist)]
            line.extend(states)
            yield line

