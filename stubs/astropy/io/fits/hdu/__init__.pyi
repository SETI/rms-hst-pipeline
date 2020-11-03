from typing import Any, Iterable, Iterator, Optional, Sequence, Sized
from astropy.io.fits.column import ColDefs

class BinTableHDU(Sized):
    columns: ColDefs
    data: Any
    def __len__(self) -> int: ...

class PrimaryHDU:
    def __init__(self, image: Any) -> None: ...
    def writeto(self, filepath: str) -> None: ...

class HDUList(Sequence[Any], Sized):
    def close(
        self,
        output_verify: str = "exception",
        verbose: bool = False,
        closed: bool = True,
    ) -> None: ...
    def __len__(self) -> int: ...
    # Here __getitem__() could also take a slice but we don't care, so we
    # tell mypy to shut up.
    def __getitem__(self, index: int) -> Any: ...  # type: ignore[override]
    def __iter__(self) -> Iterator[Any]: ...

def fitsopen(
    name: str,
    mode: str = "readonly",
    memmap: Optional[bool] = None,
    save_backup: Optional[bool] = False,
    cache: Optional[bool] = True,
    lazy_load_hdus: Optional[bool] = None,
    **kwargs: Any
) -> HDUList: ...
