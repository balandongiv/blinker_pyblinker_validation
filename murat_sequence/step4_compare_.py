

from pyblinker.utils.evaluation import (
	blink_comparison,

	)

TOLERANCE_SAMPLES = 20                   # blink start/end alignment tolerance
# load all the outputs from previous steps
detection_pyblinker # The input from pyblinker blink detection
ground_truth_events # The ground truth blink events from blinker
# 14) Compute alignment table and summary metrics (matches, differences, etc.)
alignments, metrics = blink_comparison.compute_alignments_and_metrics(
	detected_df=detection_pyblinker,
	ground_truth_df=ground_truth_events,
	tolerance_samples=TOLERANCE_SAMPLES,
	)