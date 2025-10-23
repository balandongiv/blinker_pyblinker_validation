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


# ---------- Runner ----------

def main():
    # Start MATLAB headless
    eng = matlab.engine.start_matlab("-nojvm -nosplash -nodesktop")

    # === EDIT THESE THREE PATHS ===
    eeglab_root = r"D:\code development\matlab_plugin\eeglab2025.1.0"
    project_root = r"C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\src\matlab_runner"
    edf_file = r"/src/matlab_runner/seg_annotated_raw.edf"

    # Put your MATLAB code + EEGLAB + Blinker on path
    eng.addpath(eng.genpath(project_root), nargout=0)
    eng.addpath(eeglab_root, nargout=0)
    eng.eeglab('nogui', nargout=0)  # init EEGLAB w/o GUI
    eng.addpath(eng.genpath(str(Path(eeglab_root) / "plugins" / "Blinker1.2.0")), nargout=0)

    # (Optional) sanity check
    print("MATLABROOT:", eng.eval("matlabroot", nargout=1))
    print("getBlinkerDefaults ->", eng.which("getBlinkerDefaults"))
    print("run_blinker_pipeline ->", eng.which("run_blinker_pipeline"))

    try:
        # out must be a single scalar struct returned by your wrap function
        out = eng.run_blinker_pipeline_wrap(edf_file, nargout=1)

        # Build separate DataFrames
        blinkFits   = to_dataframe(out['blinkFits'])
        blinkProps  = to_dataframe(out['blinkProps'])
        blinkStats  = to_dataframe(out['blinkStats'])
        blinks      = to_dataframe(out['blinks'])
        params      = to_dataframe(out['params'])

        # Example: show shapes
        print("blinkFits  :", blinkFits.shape)
        print("blinkProps :", blinkProps.shape)
        print("blinkStats :", blinkStats.shape)
        print("blinks     :", blinks.shape)
        print("params     :", params.shape)

        # OPTIONAL: save to CSVs next to the EDF
        out_dir = Path(edf_file).with_suffix("").parent
        blinkFits.to_csv("blinkFits.csv", index=False)
        blinkProps.to_csv("blinkProps.csv", index=False)
        blinkStats.to_csv("blinkStats.csv", index=False)
        blinks.to_csv("blinks.csv", index=False)
        params.to_csv("params.csv", index=False)

        # If you want them as variables available to other modules:
        return blinkFits, blinkProps, blinkStats, blinks, params

    except matlab.engine.MatlabExecutionError as e:
        print("MATLAB error:\n", e)
        raise
    finally:
        eng.quit()


if __name__ == "__main__":
    main()
