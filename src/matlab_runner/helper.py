import matlab.engine
from pathlib import Path
import pandas as pd
import numpy as np


# ---------- MATLAB -> Python converters ----------

def _is_matlab_struct(x):
    """MATLAB struct as seen by the Engine behaves like a mapping with .keys() and [] access."""
    return hasattr(x, "keys") and hasattr(x, "__getitem__") and not isinstance(x, dict)

def _is_matlab_cell(x):
    # cell arrays are list-like; we detect by module/type name to be robust
    tn = type(x).__name__.lower()
    return ("cell" in tn) or (hasattr(x, "__iter__") and not _is_matlab_struct(x) and not isinstance(x, (list, tuple, dict, str, bytes)))

def ml_to_py(obj):
    """
    Recursively convert MATLAB Engine values to plain Python types:
    - struct -> dict
    - cell -> list
    - numeric arrays -> numpy arrays (scalars become Python scalars)
    """
    # Primitives pass through
    if obj is None or isinstance(obj, (bool, int, float, str, bytes)):
        return obj

    # MATLAB numeric/logical arrays -> numpy
    try:
        import matlab  # type: ignore
        matlab_numeric_types = (
            matlab.double, matlab.int8, matlab.int16, matlab.int32, matlab.int64,
            matlab.uint8, matlab.uint16, matlab.uint32, matlab.uint64,
            matlab.logical,  # noqa
        )
        if isinstance(obj, matlab_numeric_types):
            arr = np.array(obj)
            if arr.ndim == 0 or (arr.size == 1):
                return arr.item()
            return arr
    except Exception:
        pass

    # Struct
    if _is_matlab_struct(obj):
        return {k: ml_to_py(obj[k]) for k in obj.keys()}

    # Cell / other iterable engine arrays
    if _is_matlab_cell(obj):
        try:
            # Try to iterate; some engine arrays support len()/indexing
            return [ml_to_py(obj[i]) for i in range(len(obj))]
        except Exception:
            return [ml_to_py(x) for x in list(obj)]

    # Native Python containers
    if isinstance(obj, dict):
        return {k: ml_to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ml_to_py(v) for v in obj]

    # Fallback: just return as-is
    return obj


def to_dataframe(x):
    """
    Turn a MATLAB value (already converted via ml_to_py or raw) into a tidy DataFrame:
    - list of dict -> rows
    - dict -> single-row DataFrame
    - everything else -> one-column DataFrame
    """
    x = ml_to_py(x)

    if isinstance(x, list) and all(isinstance(e, dict) for e in x):
        # Struct array (already made safe -> list of scalar structs)
        return pd.json_normalize(x)

    if isinstance(x, dict):
        # Single scalar struct
        return pd.json_normalize(x)

    if isinstance(x, list) and not any(isinstance(e, (dict, list, tuple, np.ndarray)) for e in x):
        # Flat list of scalars
        return pd.DataFrame({"value": x})

    # Default: keep object as a single cell
    return pd.DataFrame({"value": [x]})