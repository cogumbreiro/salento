from typing import *

class BayesianEncoder:
    psi_mean: Any
    psi_covariance: Any
    inputs: Any
    def __init__(self, config:Any) -> None: ...

class BayesianDecoder:
    edges: Any
    nodes: Any
    def __init__(self, config:Any, initial_state:Any, infer:bool) -> None: ...
