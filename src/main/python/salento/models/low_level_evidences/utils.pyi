from typing import *
import argparse
from numpy import np

CHILD_EDGE: str
SIBLING_EDGE: str

def read_config(js:Dict[str,Any], chars_vocab:np.ndarray) -> argparse.Namespace: ...
def dump_config(config:argparse.Namespace) -> Dict[str,Any]: ...
