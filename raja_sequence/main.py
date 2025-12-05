import mne
import json
import os
from helper import load_ground_truth, unzip_file, restructure_blink_dataframe


# EDIT THESE PATHS
CVAT_ROOT = r"C:\Users\balan\OneDrive - ums.edu.my\CVAT_visual_annotation\cvat_zip_final"
DATA_ROOT = r"D:\dataset\drowsy_driving_raja_processed"

def build_path_dic(cvat_root: str):
	"""
	Build a dict:
		path_dic[subject_id][session_name] = full_path_to_zip

	Example:
		path_dic['S1']['S01_20170519_043933']   = '...\\S1\\from_cvat\\S01_20170519_043933.zip'
		path_dic['S1']['S01_20170519_043933_2'] = '...\\S1\\from_cvat\\S01_20170519_043933_2.zip'
	"""
	path_dic = {}

	for subject_id in os.listdir(cvat_root):  # S1, S2, ...
		subj_root = os.path.join(cvat_root, subject_id, "from_cvat")
		if not os.path.isdir(subj_root):
			continue

		ses_dict = {}
		for fname in os.listdir(subj_root):
			if not fname.lower().endswith(".zip"):
				continue
			session_name = os.path.splitext(fname)[0]  # strip .zip
			full_zip_path = os.path.join(subj_root, fname)
			ses_dict[session_name] = full_zip_path

		if ses_dict:
			path_dic[subject_id] = ses_dict

	return path_dic


def iter_fif_files(data_root: str):
	"""
	Yield (subject_id, fif_path) for every .fif file under DATA_ROOT.
	"""
	for subject_id in os.listdir(data_root):  # S1, S2, ...
		subj_dir = os.path.join(data_root, subject_id)
		if not os.path.isdir(subj_dir):
			continue

		for root, _, files in os.walk(subj_dir):
			for fname in files:
				if fname.lower().endswith(".fif"):
					yield subject_id, os.path.join(root, fname)

def process_fif_file(fif_file, subject_id, path_dic, **kwargs):
	"""
	   Process a single FIF file together with one matching CVAT ZIP archive.

	   Expected folder layout
	   ----------------------
	   1. EEG / FIF side (DATA_ROOT):

		   D:\
		   └─ dataset\
			  └─ drowsy_driving_raja_processed\
				 ├─ S1\
				 │  ├─ S01_20170519_043933\
				 │  │  ├─ ear_eog.fif
				 │  │  └─ (other FIF / aux files ...)
				 │  └─ S01_20170519_043933_2\
				 │     ├─ ear_eog.fif
				 │     └─ ...
				 └─ S2\
					├─ TEST_20170601_042544\
					│  ├─ ear_eog.fif
					│  └─ ...
					└─ TEST_20170601_042544_2\
					   ├─ ear_eog.fif
					   └─ ...

		   In other words:
			   DATA_ROOT/
				 <subject_id>/                 # e.g. S1, S2, ...
				   <session_name>/             # e.g. S01_20170519_043933, TEST_20170601_042544_2
					 *.fif                     # e.g. ear_eog.fif

		   `fif_file` is the full path to one of these *.fif files.
		   `subject_id` is the directory name under DATA_ROOT (e.g. "S1").
		   `session_name` is derived from the folder that directly contains the FIF file.

	   2. CVAT / ZIP side (CVAT_ROOT):

		   C:\
		   └─ Users\
			  └─ balan\
				 └─ OneDrive - ums.edu.my\
					└─ CVAT_visual_annotation\
					   └─ cvat_zip_final\
						  ├─ S1\
						  │  └─ from_cvat\
						  │     ├─ S01_20170519_043933.zip
						  │     ├─ S01_20170519_043933_2.zip
						  │     └─ S01_20170519_043933_3.zip
						  └─ S2\
							 └─ from_cvat\
								├─ TEST_20170601_042544.zip
								├─ TEST_20170601_042544_2.zip
								└─ TEST_20170601_042544_3.zip

		   In other words:
			   CVAT_ROOT/
				 <subject_id>/                 # e.g. S1, S2, ...
				   from_cvat/
					 <session_name>.zip        # e.g. S01_20170519_043933.zip

		   `path_dic` is built such that:
			   path_dic[subject_id][session_name] == full path to the corresponding .zip

	   Function behavior
	   -----------------
	   Given:
		   fif_file  -> one EEG FIF file (ear_eog.fif)
		   subject_id -> e.g. "S1"
		   path_dic  -> mapping of subject/session to CVAT zip path

	   The function will:
		   1. Derive `session_name` from the parent folder of `fif_file`.
		   2. Look up the corresponding ZIP path:
				  zip_path = path_dic[subject_id][session_name]
		   3. Unzip that single archive into a folder under its ZIP directory.
		   4. Find `default-annotations-human-imagelabels.csv` inside the extracted content.
		   5. Load the time-shift from config_video_detail.json (shift_cvat[subject_id][session_name]).
		   6. Convert CVAT ImageIDs to seconds using the given sampling rate.
		   7. Create an mne.Annotations object and attach it to the FIF.
		   8. Plot the annotated FIF.

	   Parameters
	   ----------
	   fif_file : str
		   Full path to the .fif file.
	   subject_id : str
		   Subject ID (e.g. "S1").
	   path_dic : dict
		   Dictionary mapping path_dic[subject_id][session_name] -> zip_path.
	   kwargs :
		   sampling_rate : float, optional
			   Video frame rate used for ImageID → time conversion (default 30.0).
		   event_duration : float, optional
			   Duration in seconds for each annotation (default 1.0).

	   Returns
	   -------
	   raw : mne.io.Raw | None
		   The loaded Raw object with annotations attached, or None if skipped.
	   annotations : mne.Annotations | None
		   The annotations object, or None if skipped.
	   df_gt : pandas.DataFrame | None
		   Ground-truth dataframe containing the CVAT labels, or None if skipped.
	   """
	sampling_rate = float(kwargs.get("sampling_rate", 30.0))
	zip_path = kwargs["zip_path"]          # one zip per call
	session_name = kwargs["session_name"]  # folder name above fif

	base_dir = os.path.dirname(zip_path)
	extract_folder = os.path.join(base_dir, session_name)
	os.makedirs(extract_folder, exist_ok=True)

	unzip_file(zip_path, extract_folder)
	# ---- locate the ground-truth CSV --------------------------------------
	gt_csv_path = None
	for root, _, files in os.walk(extract_folder):
		for fname in files:
			if fname == "default-annotations-human-imagelabels.csv":
				gt_csv_path = os.path.join(root, fname)
				break
		if gt_csv_path is not None:
			break

	if gt_csv_path is None:
		print(f"[WARN] No 'default-annotations-human-imagelabels.csv' found in {extract_folder}")
		return

	print(f"[INFO] Using ground-truth CSV: {gt_csv_path}")

	# ---- load shift from config_video_detail.json --------------------------
	json_file = os.path.join(os.path.dirname(__file__), "..", "config_video_detail.json")
	json_file = os.path.normpath(json_file)

	with open(json_file, "r") as f:
		data_video = json.load(f)

	try:
		# assumes shift_cvat keys are session names like "S01_20170519_043933_2"
		constant_shift = data_video["data"][subject_id]["shift_cvat"][session_name]
	except KeyError:
		print(f"[WARN] No shift_cvat entry for {subject_id}/{session_name}; using 0")
		constant_shift = 0

	print(f"[INFO] constant_shift = {constant_shift}")

	# ---- load ground truth & convert to seconds ----------------------------
	df_gt = load_ground_truth(gt_csv_path, constant_shift, sampling_rate)
	sampling_rate = 30.0  # frames per second
	df_gt = restructure_blink_dataframe(df_gt, sampling_rate)
	# (Optional) keep only labels ending with "_min"
	# df_gt = filter_min_labels(df_gt)

	if df_gt.empty:
		print("[WARN] Ground-truth dataframe is empty; nothing to annotate.")
		return

	# ---- convert df_gt to MNE Annotations ----------------------------------
	# compute start_sec from start (may still be negative at this point)
	df_gt["start_sec"] = df_gt["start"] / sampling_rate

	# --- Find negatives ------------------------------------------------------
	neg_onset_mask = df_gt["start_sec"] < 0
	neg_duration_mask = df_gt["duration_seconds"] < 0

	neg_onset_idx = df_gt.index[neg_onset_mask].tolist()
	neg_duration_idx = df_gt.index[neg_duration_mask].tolist()

	if neg_onset_idx:
		print(f"[WARN] Negative onset values found at rows: {neg_onset_idx}")
		print(df_gt.loc[neg_onset_idx,
		["blink_type", "start", "start_sec", "remark"]])

	if neg_duration_idx:
		print(f"[WARN] Negative duration values found at rows: {neg_duration_idx}")
		print(df_gt.loc[neg_duration_idx,
		["blink_type", "start", "min", "end",
		 "duration_seconds", "remark"]])

		# prepend remark text and change blink_type for those rows
		df_gt.loc[neg_duration_mask, "remark"] = (
				"duration is negative before abs; "
				+ df_gt.loc[neg_duration_mask, "remark"].fillna("")
		)
		df_gt.loc[neg_duration_mask, "blink_type"] = "blink_negative_duration"

	# --- Fix negatives (make them positive) ----------------------------------
	df_gt["start_sec"] = df_gt["start_sec"].abs()
	df_gt["duration_seconds"] = df_gt["duration_seconds"].abs()

	# --- Build arrays for MNE Annotations ------------------------------------
	onset = df_gt["start_sec"].to_numpy(dtype=float)
	description = df_gt["blink_type"].astype(str).tolist()
	duration = df_gt["duration_seconds"].to_numpy(dtype=float)



	annotations = mne.Annotations(
			onset=onset,
			duration=duration,
			description=description
			)

	# ---- load FIF, attach annotations, and plot ----------------------------
	raw = mne.io.read_raw_fif(fif_file, preload=True)
	raw.set_annotations(annotations)

	title = f"{subject_id} | {session_name} | {session_name}"
	print(f"[INFO] Plotting: {title}")
	raw.plot(block=True, title=title)

	# Optionally return for downstream analysis
	return raw, annotations, df_gt

def main():
	# Build: path_dic[subject_id][session_name] = zip_path
	path_dic = build_path_dic(CVAT_ROOT)

	# Walk all FIF files
	for subject_id, fif_path in iter_fif_files(DATA_ROOT):
		session_name = os.path.basename(os.path.dirname(fif_path))   # e.g. "S01_20170519_043933_2"

		# Check if we have a matching zip for this (subject, session)
		zip_path = None
		if subject_id in path_dic:
			zip_path = path_dic[subject_id].get(session_name)

		if zip_path is None:
			print(f"[WARN] No zip found for {subject_id}/{session_name}; skipping {fif_path}")
			continue

		print(f"\n=== Processing FIF: {fif_path}")
		print(f"    using ZIP: {zip_path}")

		# pass zip_path + session_name via kwargs (one fif + one zip)
		process_fif_file(
			fif_file=fif_path,
			subject_id=subject_id,
			path_dic=path_dic,
			zip_path=zip_path,
			session_name=session_name,
			)
if __name__ == "__main__":
	main()


