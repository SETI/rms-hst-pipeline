# Dumb stubs to shut mypy up.  There is reportedly work to develop
# mypy stubs for numpy.  When that happens, add them in.  Currently
# (mid-2020), numpy is only used by pdart to remove NaNs from image
# data.

from typing import Any

ndarray: Any
isnan: Any
sum: Any
