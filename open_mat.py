from scipy.io import loadmat
import numpy as np
import mne

mat_path = r"C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\s01.mat"

# 1) Load MAT in a way that gives nice Python objects for structs
mat = loadmat(mat_path, struct_as_record=False, squeeze_me=True)

# 2) Pull the EEG struct
eeg = mat.get('eeg')
if eeg is None:
    raise RuntimeError("No 'eeg' struct in MAT file.")

# 3) Choose which field contains continuous (or concatenatable) data
#    Common options your file seems to have: 'rest', 'noise', 'movement_left',
#    'movement_right', 'imagery_left', 'imagery_right', etc.
#    Pick ONE to make a Raw. Start with 'rest' if present:
preferred_fields = ['rest', 'noise', 'movement_left', 'movement_right', 'imagery_left', 'imagery_right']
data_block = None
for key in preferred_fields:
    if hasattr(eeg, key):
        data_block = getattr(eeg, key)
        which = key
        break

if data_block is None:
    raise RuntimeError("Couldn't find any of the expected data fields: " + ", ".join(preferred_fields))

# 4) Convert the chosen field to a (n_channels, n_times) float array
def to_2d_numeric(arr):
    """
    Accepts:
      - 2D numeric array (n_channels, n_times) -> returns as float
      - list/tuple of 2D trials -> concatenates along time
      - object arrays from MATLAB cell arrays -> flattens/concats
      - 3D arrays (n_trials, n_channels, n_times) -> concat trials along time
    Returns: 2D float array (n_channels, n_times)
    """
    a = arr

    # If it's a list/tuple of arrays (e.g., trials), concat along time
    if isinstance(a, (list, tuple)):
        a = [np.asarray(x) for x in a]
        # Assume each x is (n_channels, n_times) or (n_times, n_channels)
        # First standardize orientation, then concat along time
        fixed = []
        for x in a:
            if x.ndim != 2:
                raise ValueError(f"Trial has shape {x.shape}; expected 2D.")
            # Heuristic: times >> channels, so if channels < times we want (n_channels, n_times)
            x = x if x.shape[0] > x.shape[1] else x.T
            fixed.append(x.astype(float, copy=False))
        return np.concatenate(fixed, axis=1)

    # If it's an object array (common for MATLAB cell arrays)
    if isinstance(a, np.ndarray) and a.dtype == object:
        elems = [np.asarray(x) for x in a.ravel().tolist()]
        fixed = []
        for x in elems:
            if x.ndim != 2:
                raise ValueError(f"Cell element has shape {x.shape}; expected 2D.")
            x = x if x.shape[0] > x.shape[1] else x.T
            fixed.append(x.astype(float, copy=False))
        return np.concatenate(fixed, axis=1)

    # If it's a numpy numeric array already
    a = np.asarray(a)
    if a.ndim == 3:
        # Likely (n_trials, n_channels, n_times) -> concat trials along time
        if a.shape[1] < a.shape[2]:
            # ensure channels axis is 1
            pass
        # reshape to (n_channels, n_trials*n_times)
        a = np.concatenate([a[i] for i in range(a.shape[0])], axis=1)  # WRONG axis if (trials, ch, time)
        # Let's do this cleanly:
        # a is (trials, ch, time) OR (ch, time, trials) OR (time, ch, trials)
        # Try to detect (trials, ch, time):
        if a.ndim == 3:
            if a.shape[0] < a.shape[1] and a.shape[1] < a.shape[2]:
                # assume (trials, ch, time)
                a = np.concatenate([a[t] for t in range(a.shape[0])], axis=1)
            elif a.shape[1] < a.shape[2] and a.shape[0] < a.shape[2]:
                # try (ch, time, trials)
                a = np.concatenate([a[:, :, t] for t in range(a.shape[2])], axis=1)
            else:
                raise ValueError(f"Don't know how to handle 3D shape {a.shape}; please clarify axes.")
    if a.ndim != 2:
        raise ValueError(f"Expected 2D after conversion, got shape {a.shape}")

    # Ensure (n_channels, n_times)
    a = a if a.shape[0] > a.shape[1] else a.T
    return a.astype(float, copy=False)

data = to_2d_numeric(data_block)

# 5) Sampling rate
sfreq = None
for k in ('srate', 'fs', 'sfreq', 'Fs'):
    if hasattr(eeg, k):
        sfreq = float(np.asarray(getattr(eeg, k)).squeeze())
        break
if sfreq is None:
    # fallback (you can hardcode if you know it)
    raise RuntimeError("Sampling rate not found in struct (looked for 'srate', 'fs', 'sfreq', 'Fs').")

# 6) Channel names (try to extract, else generic)
n_channels, n_times = data.shape
ch_names = [f'ch{i+1}' for i in range(n_channels)]

# If labels exist (often in eeg.senloc / psenloc), try to use them
for label_container in ('senloc', 'psenloc'):
    if hasattr(eeg, label_container):
        labels = getattr(eeg, label_container)
        # labels might be a struct/array of structs with a 'label' field
        try:
            if isinstance(labels, (list, tuple, np.ndarray)):
                names = []
                for item in np.ravel(labels):
                    name = None
                    # struct-like?
                    if hasattr(item, 'label'):
                        name = item.label
                    elif isinstance(item, dict) and 'label' in item:
                        name = item['label']
                    if name is not None:
                        names.append(str(np.asarray(name).squeeze()))
                if len(names) == n_channels:
                    ch_names = names
                    break
        except Exception:
            pass

# 7) Build Raw
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['eeg'] * n_channels)
raw = mne.io.RawArray(data, info)

print(raw)
print(f"Used field: '{which}', sfreq={sfreq} Hz, shape={data.shape}")

# Optional: create events/annotations if present (movement_event, imagery_event, frame)
# Example if 'movement_event' is a vector of sample indices:
if hasattr(eeg, 'movement_event'):
    mv = np.asarray(getattr(eeg, 'movement_event')).squeeze()
    if mv.ndim == 1 and mv.size > 0:
        onsets = mv / sfreq
        desc = ['movement'] * mv.size
        raw.set_annotations(mne.Annotations(onset=onsets, duration=[0]*mv.size, description=desc))

raw.plot(block=True)
# raw.save(r"C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\s01_raw.fif", overwrite=True)
