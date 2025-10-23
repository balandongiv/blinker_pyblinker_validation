import mne

# Path to your FIF file
path = r'D:\dataset\drowsy_driving_raja_processed\S1\S01_20170519_043933\seg_annotated_raw.fif'

# Load the FIF file
raw = mne.io.read_raw_fif(path, preload=True)

# --- Clean EDF-incompatible metadata ---
# EDF format does not allow spaces or long strings in certain header fields
if 'device_info' in raw.info and isinstance(raw.info['device_info'], dict):
    for key, value in raw.info['device_info'].items():
        if isinstance(value, str):
            raw.info['device_info'][key] = value.replace(' ', '_')

if 'subject_info' in raw.info and isinstance(raw.info['subject_info'], dict):
    for key, value in raw.info['subject_info'].items():
        if isinstance(value, str):
            raw.info['subject_info'][key] = value.replace(' ', '_')

# --- Export to EDF ---
try:
    raw.export("seg_annotated_raw.edf", fmt='edf')
    print("✅ File successfully saved as seg_annotated_raw.edf")
except Exception as e:
    print(f"❌ Export failed: {e}")
