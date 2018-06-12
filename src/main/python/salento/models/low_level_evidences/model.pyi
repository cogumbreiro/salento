from typing import *

import tensorflow as tf
import numpy as np

class Row(NamedTuple):
    node: str
    edge: str
    distribution: np.ndarray
    state: Any
    cache_id: str

class Model:

    def __init__(self, config:Any, infer:bool=False) -> None: ...

    def infer_psi(self, sess:tf.Session, evidences:Any) -> Any: ...

    def infer_seq(self,
        sess:tf.Session,
        psi:Any,
        seq:Iterable[Tuple[str,str]],
        cache:Optional[Dict[str,Any]]=None,
        resume:Optional[Row]=None) -> np.ndarray: ...

    def infer_seq_iter(self, sess:tf.Session, psi:Any, seq:Iterable[Tuple[str,str]], cache:Optional[Dict[str,Any]]=None, resume:Optional[Row]=None) -> Iterable[Row]: ...

    def _infer_seq_step(self, sess:tf.Session, state:Any, node:str, edge:str) -> Tuple[np.ndarray, Any]: ...
