import matlab.engine
from pathlib import Path


from src.matlab_runner.helper import to_dataframe
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
