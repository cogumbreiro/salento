from typing import *

import tensorflow as tf
import numpy as np
from salento.models.low_level_evidences.model import Model

from salento.models.low_level_evidences import model

# Types used in the class
# A term is a string
Term = str
# The probability distribution of each term
TermDist = Dict[Term, float]
# The cache used in the model
Cache = Dict[str, Any]
# A JSON call-event
Event = Dict[str, Any]

class Entry(NamedTuple):
    """
    Pairs a term with a term distribution probability.
    Can be used a `Tuple[Term, TermDist]`.
    """
    term: Term
    distribution: TermDist

def event_states(call:Event) -> Iterable[Term]: ...
def _next_state(event:Event) -> Iterable[Tuple[str,str]]: ...
def _next_call(event:Event) -> Tuple[str, str]: ...
def _sequence_to_graph(sequence:List[Event], step:str) -> List[Tuple[str,str]]: ...
def _call_names(terms:List[Event], sentinel:Term) -> Iterable[Term]: ...
class VectorMapping(TermDist):
    def __init__(self, data:List[float], id_to_term:List[Term], term_to_id:Mapping[Term,int]) -> None: ...
    def keys(self) -> KeysView[Term]: ...
    #def __iter__(self) -> Iterable[Term]: ...
    def values(self) -> ValuesView[float]: ...
    def __contains__(self, key:Any) -> bool: ...
    #def get(self, key, default:Optional[float]) -> Optional[float]: ...
    #def items(self) -> Iterable[Tuple[Term,float]]: ...
    def __getitem__(self, key:Term) -> float: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

T = TypeVar('T')
def iter_append(elems:Iterable[T], elem:T) -> Iterable[T]: ...

class BayesianPredictor:
    @classmethod
    def load(cls:Type['BayesianPredictor'], save:str, sess:tf.Session) -> 'BayesianPredictor': ...
    def __init__(self, model:Model, sess:tf.Session) -> None: ...
    def restore(self, save:str) -> None: ...
    def infer_step(self, psi:Any, sequence:List[Event], step:str, cache:Optional[Cache]=None) -> TermDist: ...
    def _create_distribution(self, dist:List[float]) -> TermDist: ...
    def psi_random(self) -> np.ndarray: ...
    def infer_call_iter(self, psi:Any, sequence:List[Event], sentinel:Term, cache:Optional[Cache]=None) \
            -> Iterable[Entry]: ...
    def infer_state_iter(self, psi:Any, sequence:List[Event], sentinel:Term, cache:Optional[Cache]=None) \
            -> Iterable[List[Entry]]: ...

