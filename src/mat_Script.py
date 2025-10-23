# run_pipeline.py
import matlab.engine
from pathlib import Path

def main():
    # Start MATLAB headless
    eng = matlab.engine.start_matlab("-nojvm -nosplash -nodesktop")

    # === EDIT THESE THREE PATHS ===
    eeglab_root = r"D:\code development\matlab_plugin\eeglab2025.1.0"
    project_root = r"C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\src\matlab_runner"
    edf_file = r"C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\src\matlab_runner\seg_annotated_raw.edf"

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
        out = eng.run_blinker_pipeline_wrap(edf_file, nargout=1)
        # unpack from the cell array
        blinks, blinkFits, blinkProps, blinkStats, params, com = out
        print("Pipeline OK. com =", out['com'])

        # Example: read a few fields
        blinks = out['blinks']
        print("Blinker status:", blinks.get('status', 'unknown'))

        # If you need lengths of struct arrays
        # (Engine exposes them as sequence-like; len(...) usually works)
        blinkFits = out['blinkFits']
        print("blinkFits count:", len(blinkFits))

        blinkProps = out['blinkProps']
        print("blinkProps count:", len(blinkProps))

    except matlab.engine.MatlabExecutionError as e:
        print("MATLAB error:\n", e)
        raise
    finally:
        eng.quit()

if __name__ == "__main__":
    main()
