from typing import *

import tensorflow as tf
import numpy as np
from salento.models.low_level_evidences.architecture import BayesianEncoder, BayesianDecoder

class Row(NamedTuple):
    node: str
    edge: str
    distribution: List[float]
    state: Any
    cache_id: Optional[str]

class Model:
    encoder: BayesianEncoder
    decoder: BayesianDecoder
    loss: Any
    evidence_loss: Any
    latent_loss: Any
    train_op: Any
    gen_loss: Any
    targets: Any

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
