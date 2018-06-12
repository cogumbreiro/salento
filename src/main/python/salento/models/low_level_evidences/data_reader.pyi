from typing import *

LOADERS: Dict[str, Callable[[str], IO]]

def smart_open(filename:str, *args:Any, **kwargs:Any) -> IO: ...
def get_seq_paths(js:List[Dict[str,Any]]) -> List[Tuple[str,str]]: ...

class Reader:
    reset_batches: Any

    def __init__(self, clargs:Any, config:Any) -> None: ...

    def next_batch(self) -> Tuple[Any,Any,Any,Any]: ...
