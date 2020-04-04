from typing import Optional, Sized

class HDUList(Sized):
    def close(
        self,
        output_verify: str = "exception",
        verbose: bool = False,
        closed: bool = True,
    ) -> None: ...
    def __len__(self) -> int: ...

def fitsopen(
    name: str,
    mode: str = "readonly",
    memmap: Optional[bool] = None,
    save_backup: Optional[bool] = False,
    cache: Optional[bool] = True,
    lazy_load_hdus: Optional[bool] = None,
    **kwargs
) -> HDUList: ...
