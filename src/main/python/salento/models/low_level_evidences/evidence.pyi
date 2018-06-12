from typing import *

import tensorflow as tf
import numpy as np

class Evidence:

    def init_config(self, evidence:Any, chars_vocab:np.ndarray) -> None: ...
    def dump_config(self) -> Dict[str,Any]: ...

