from __future__ import annotations
from typing import Any, Union, Dict, List

import numpy as np
import numpy.typing as npt

_DetectionDataValueType = Union[npt.NDArray[np.generic], List[Any]]
_DetectionDataType = Dict[str, _DetectionDataValueType]
_MetadataType = Dict[str, Any]
