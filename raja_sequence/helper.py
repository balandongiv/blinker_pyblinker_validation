
import numpy as np
import pandas as pd

import mne

import os

import warnings


def restructure_blink_dataframe(
		df: pd.DataFrame,
		sampling_rate: float,
		frame_col: str = "adjusted_frame",
		label_col: str = "LabelName",
		) -> pd.DataFrame:
	"""
	Restructure a CVAT-style blink dataframe into one row per blink.

	Parameters
	----------
	df : pandas.DataFrame
		Input dataframe with (at least) the columns:

			- ``LabelName`` : str
				  Contains labels like:
					  ``HB_CL_left_start``
					  ``HB_CL_min``
					  ``HB_CL_right_end``
					  ``HB_M_left_start`` ...
			- ``adjusted_frame`` : int
				  Frame index (after time shift).

		The dataframe is assumed to be in chronological order
		(increasing ``adjusted_frame``). If not, it will be sorted.

	sampling_rate : float
		Video frame rate (frames per second). Used to convert the
		start–end frame difference into a duration in seconds:

			``duration_seconds = (end - start) / sampling_rate``.

	frame_col : str, default ``"adjusted_frame"``
		Name of the column containing frame numbers.

	label_col : str, default ``"LabelName"``
		Name of the column containing label strings.

	Returns
	-------
	out : pandas.DataFrame
		A new dataframe with one row per blink and columns:

			- ``blink_type``        : str (e.g. ``"HB_CL"``, ``"B_CL"``)
			- ``start``             : int, frame index of blink start
			- ``min``               : int, frame index of minimum
			- ``end``               : int, frame index of blink end
			- ``duration_frames``   : int, ``end - start``
			- ``duration_seconds``  : float, ``duration_frames / sampling_rate``

	Blink type and phase parsing
	----------------------------
	* ``blink_type`` is defined as the first two underscore-separated
	  tokens of ``LabelName``:

		  - ``"HB_CL_left_start"``  ->  blink_type = ``"HB_CL"``
		  - ``"HB_CL_min"``         ->  blink_type = ``"HB_CL"``
		  - ``"HB_M_right_end"``    ->  blink_type = ``"HB_M"``
		  - ``"B_CL_min"``          ->  blink_type = ``"B_CL"``

	* The **phase** is taken from the last token in ``LabelName``:

		  - ``"..._start"``  -> phase = ``"start"``
		  - ``"..._min"``    -> phase = ``"min"``
		  - ``"..._end"``    -> phase = ``"end"``

	Missing phase rules
	-------------------
	For each blink we collect at most one value for each phase:
	``start``, ``min``, ``end``. When we finalize a blink:

	* If **start is missing**, use the ``min`` value.
	* If **min is missing**, use the ``start`` value.
	* If **end is missing**, use the ``min`` value.
	* If **two or more** of the three phases are missing,
	  a warning is raised and that blink is **skipped**.

	Examples
	--------
	1. Perfect triple (no missing phases)

	   Original rows:

	   ==========  ===================  ================
	   LabelName   adjusted_frame       Comment
	   ----------  -----------------    ----------------
	   HB_CL_left_start    353          start
	   HB_CL_min           356          min
	   HB_CL_right_end     361          end
	   ==========  ===================  ================

	   Output row:

	   - blink_type       = ``"HB_CL"``
	   - start            = 353
	   - min              = 356
	   - end              = 361
	   - duration_frames  = 361 - 353 = 8
	   - duration_seconds = 8 / sampling_rate

	2. Missing start

	   Original rows:

	   - ``HB_CL_min``           at frame 356
	   - ``HB_CL_right_end``     at frame 361

	   There is no explicit ``*_start``. We apply:

	   - start := min = 356
	   - min   = 356
	   - end   = 361

	3. Missing min

	   Original rows:

	   - ``HB_CL_left_start``    at frame 353
	   - ``HB_CL_right_end``     at frame 361

	   We apply:

	   - start = 353
	   - min   := start = 353
	   - end   = 361

	4. Missing end

	   Original rows:

	   - ``HB_CL_left_start``    at frame 353
	   - ``HB_CL_min``           at frame 356

	   We apply:

	   - start = 353
	   - min   = 356
	   - end   := min = 356

	5. Two or more phases missing

	   Examples:

	   * Only ``HB_CL_min`` with no corresponding start or end.
	   * Only ``HB_CL_left_start`` with no min or end.

	   In these cases a warning is emitted via ``warnings.warn``,
	   and the blink is not included in the output.

	Notes
	-----
	* The function scans the dataframe row by row (time order) and
	  groups phases into blinks in the order they appear.
	* If you have interleaved blink types, each blink type is handled
	  separately in sequence; when a blink is complete or a new
	  ``*_start`` for the same blink_type appears, a new blink group
	  is started.
	"""
	# Work on a copy & enforce time order
	df = df.copy()
	df = df.sort_values(frame_col).reset_index(drop=True)

	def _blink_type(label: str) -> str:
		parts = str(label).split("_")
		return "_".join(parts[:2]) if len(parts) >= 2 else str(label)

	def _phase(label: str) -> str:
		return str(label).split("_")[-1]

	df["blink_type"] = df[label_col].apply(_blink_type)
	df["phase"] = df[label_col].apply(_phase)

	events = []

	def finalize_event(ev):
		"""Apply missing rules, compute duration, and append to events."""
		orig_s, orig_m, orig_e = ev["start"], ev["min"], ev["end"]
		missing_flags = {
				"start": orig_s is None,
				"min": orig_m is None,
				"end": orig_e is None,
				}
		missing_count = sum(missing_flags.values())

		# If two or more phases are missing: warn and skip
		if missing_count >= 2:
			warnings.warn(
				f"Skipping blink for type {ev['blink_type']} – "
				f"too many missing phases (start={orig_s}, min={orig_m}, end={orig_e})."
				)
			return

		# Impute as needed; keep track for remark/remark_code
		s, m, e = orig_s, orig_m, orig_e
		remark = ""
		remark_code = 0

		if missing_count == 0:
			remark_code = 0
			remark = "complete start/min/end; no imputation"
		else:
			# Exactly one missing; figure out which
			if missing_flags["start"]:
				# start missing → use min (if exists) else end (shouldn't be None here)
				s = m if m is not None else e
				remark_code = 1
				remark = "start imputed from min"
			elif missing_flags["min"]:
				# min missing → use start
				m = s
				remark_code = 2
				remark = "min imputed from start"
			elif missing_flags["end"]:
				# end missing → use min
				e = m
				remark_code = 3
				remark = "end imputed from min"

		duration_frames = e - s
		duration_seconds = duration_frames / float(sampling_rate)

		events.append(
			{
					"blink_type": ev["blink_type"],
					"start": int(s),
					"min": int(m),
					"end": int(e),
					"duration_frames": int(duration_frames),
					"duration_seconds": float(duration_seconds),
					"remark_code": int(remark_code),
					"remark": remark,
					}
			)

	current = None

	for _, row in df.iterrows():
		btype = row["blink_type"]
		phase = row["phase"]
		frame = row[frame_col]

		# Initialize first event
		if current is None:
			current = {"blink_type": btype, "start": None, "min": None, "end": None}

		# If blink type changed or current event is already full, finalize & start new
		if (
				btype != current["blink_type"]
				or (current["start"] is not None and current["min"] is not None and current["end"] is not None)
		):
			finalize_event(current)
			current = {"blink_type": btype, "start": None, "min": None, "end": None}

		# If this phase slot already filled, treat as a new blink
		if current[phase] is not None:
			finalize_event(current)
			current = {"blink_type": btype, "start": None, "min": None, "end": None}

		current[phase] = frame

	# Final leftover event
	if current is not None and any(current[p] is not None for p in ("start", "min", "end")):
		finalize_event(current)

	return pd.DataFrame(events)
def load_ground_truth(csv_path: str, constant_shift: int, sampling_rate: float) -> pd.DataFrame:
	"""Load ground truth and compute time in seconds."""
	df_gt = pd.read_csv(csv_path)
	df_gt["framenumber"] = df_gt["ImageID"].str.extract(r"(\d+)").astype(int)
	df_gt["adjusted_frame"] = df_gt["framenumber"] - constant_shift
	df_gt["seconds"] = df_gt["adjusted_frame"] / sampling_rate
	return df_gt

def unzip_file(zip_path, extract_to):
	"""Unzips a file to the specified directory."""
	import zipfile
	if os.path.exists(zip_path):
		with zipfile.ZipFile(zip_path, 'r') as zip_ref:
			zip_ref.extractall(extract_to)
		print(f"Extracted: {zip_path} -> {extract_to}")
	else:
		print(f"File not found: {zip_path}")

def load_actual_annotations(eog_path: str, preload: bool = True) -> pd.DataFrame:
	"""Load the actual annotations from an MNE .fif file."""
	raw = mne.io.read_raw_fif(eog_path, preload=preload)
	return pd.DataFrame(raw.annotations)

def filter_min_labels(df_gt: pd.DataFrame) -> pd.DataFrame:
	"""Filter ground truth to keep only rows where LabelName ends with '_min'."""
	return df_gt[df_gt["LabelName"].str.endswith("_min")].copy()


def match_ground_truth_to_annotations(df_gtruth: pd.DataFrame, df_model: pd.DataFrame) -> pd.DataFrame:
	"""
	Matches each ground truth frame in df_gtruth["adjusted_frame"] to an annotation in df_model
	based on whether it falls within the interval [frame_start, frame_end].

	If multiple matches exist, only the first unmatched annotation is assigned.
	The entire row of the matched annotation is appended to df_gtruth.

	Parameters:
		df_gtruth (pd.DataFrame): DataFrame containing ground truth events with a column "adjusted_frame".
		df_model (pd.DataFrame): DataFrame containing annotations with "frame_start" and "frame_end" columns.

	Returns:
		pd.DataFrame: Merged DataFrame of ground truth with appended annotation data.
	"""
	# Ensure we work with a copy and include an identifier for each model annotation.
	df_model = df_model.copy()
	if 'model_index' not in df_model.columns:
		df_model['model_index'] = df_model.index

	used = np.zeros(len(df_model), dtype=bool)
	starts = df_model["frame_start"].values
	ends = df_model["frame_end"].values
	model_records = df_model.to_dict('records')
	matched_annotations = []

	# Iterate over each ground truth event and find the first available matching annotation.
	for gt_time in df_gtruth["adjusted_frame"]:
		mask = (starts <= gt_time) & (ends >= gt_time)
		candidate_indices = np.flatnonzero(mask)
		match = None
		for idx in candidate_indices:
			if not used[idx]:
				used[idx] = True
				match = model_records[idx]
				break
		if match is None:
			# If no annotation is found, fill in with None values for each model column.
			match = {col: None for col in df_model.columns}
		matched_annotations.append(match)

	df_matches = pd.DataFrame(matched_annotations)
	# Concatenate ground truth with the matched annotation data.
	return pd.concat([df_gtruth.reset_index(drop=True), df_matches.reset_index(drop=True)], axis=1)
