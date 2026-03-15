
from __future__ import annotations

import logging
import math

from dataclasses import dataclass

from typing import  Mapping, Sequence

import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RecordingComparison:
	"""Container for per-recording comparison results."""

	recording_id: str
	py_events: pd.DataFrame
	blinker_events: pd.DataFrame
	metrics: Mapping[str, float]
	# onset_mae_sec: float | None


def build_summary_frame(comparisons: Sequence[RecordingComparison]) -> pd.DataFrame:
	"""Compute summary metrics describing alignment quality.

	Metrics are derived directly from ``diff_table`` (typically produced by
	:func:`pyblinker.utils.evaluation.reporting.make_diff_table`) and include:

	``total_ground_truth``
		Number of ground truth events.
	``total_detected``
		Number of detected events.
	``ground_truth_only``
		Ground truth events without a detected counterpart.
	``detected_only``
		Detected events without a ground truth counterpart.
	``share_within_tolerance``
		Count of unique events (detected plus ground truth) participating in
		amplitude- and overlap-satisfying pairs.
	``matches_within_tolerance``
		Count of unique events belonging to pairs that met the tolerance window
		but failed at least one amplitude/overlap requirement.
	``pairs_outside_tolerance``
		Count of unique events in pairs whose boundaries exceeded the tolerance
		window regardless of amplitude/overlap success.
	``share_within_tolerance_percent``
		Percentage of unique events that participate in amplitude- and overlap-
		satisfying pairs.

	``diff_table`` must include ``match_category`` and ``within_tolerance``
	columns describing how each paired event was classified:

	``"share_within_tolerance"``
		Assigned when both amplitude and overlap conditions are satisfied for a
		detected/ground-truth pair. ``within_tolerance`` may be either ``True``
		or ``False`` depending on whether the start/end boundaries also fall
		within the tolerance window.
	``"matches_within_tolerance"``
		Assigned to pairs whose start and end indices fall within the tolerance
		window but fail at least one amplitude/overlap requirement. These pairs
		are within the boundary window (``within_tolerance=True``) yet are not
		counted toward ``share_within_tolerance`` because the quality checks did
		not pass.
	``"pairs_outside_tolerance"``
		Assigned when a paired event violates the tolerance window
		(``within_tolerance=False``), even if amplitude/overlap conditions are
		otherwise satisfied.

	The ``within_tolerance`` column is a boolean flag indicating whether both
	start and end differences for a paired event fall within the configured
	tolerance. Rows without a detected/ground-truth pairing use ``NaN`` for both
	``match_category`` and ``within_tolerance``.

	Example
	-------
	Imagine ``tolerance_samples`` is ``1`` with three ground truth blinks
	(``G1``-``G3``) and three detected blinks (``D1``-``D3``). Suppose ``G1``
	and ``D1`` overlap and have similar amplitudes, so they receive
	``match_category="share_within_tolerance"`` with ``within_tolerance=True``.
	``G2`` and ``D2`` overlap but the detected start is two samples early, so
	they receive ``match_category="pairs_outside_tolerance"`` with
	``within_tolerance=False`` even though amplitude checks pass. ``G3`` and
	``D3`` align within the tolerance window but the amplitudes differ, leading
	to ``match_category="matches_within_tolerance"`` with
	``within_tolerance=True``. The resulting metrics would be:

	* ``total_ground_truth`` = 3 and ``total_detected`` = 3.
	* ``ground_truth_only`` = 0 and ``detected_only`` = 0 because all events are
	  paired.
	* ``share_within_tolerance`` = 2 because only ``G1`` and ``D1`` satisfy both
	  amplitude and overlap checks; ``share_within_tolerance_percent`` is
	  therefore ``2 / 6 * 100`` when measured against the six unique events.
	* ``matches_within_tolerance`` = 2 for the ``G3``/``D3`` pair whose
	  amplitudes differ despite boundary agreement.
	* ``pairs_outside_tolerance`` = 2 for the ``G2``/``D2`` pair whose boundaries
	  violate the tolerance window.
	"""

	def _core_metrics(tp: float, fp: float, fn: float) -> tuple[float, float, float, float]:
		precision = tp / (tp + fp) if (tp + fp) else math.nan
		recall = tp / (tp + fn) if (tp + fn) else math.nan
		if math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0:
			f1 = math.nan
		else:
			f1 = 2 * precision * recall / (precision + recall)

		accuracy = tp / (tp + fp + fn) if (tp + fp + fn) else math.nan
		return precision, recall, f1, accuracy

	rows: list[dict] = []
	for item in comparisons:
		metrics = item.metrics

		total_ground_truth = metrics.get("total_ground_truth", float(len(item.blinker_events)))
		total_detected = metrics.get("total_detected", float(len(item.py_events)))
		ground_truth_only = metrics.get("ground_truth_only", 0.0)
		detected_only = metrics.get("detected_only", 0.0)
		share_within_tolerance = metrics.get("share_within_tolerance", 0.0)
		matches_within_tolerance = metrics.get("matches_within_tolerance", 0.0)
		pairs_outside_tolerance = metrics.get("pairs_outside_tolerance", 0.0)
		unique_total = metrics.get("unique_total")
		if unique_total is None:
			unique_total = total_ground_truth + total_detected

		share_within_tolerance_percent = (
				share_within_tolerance / unique_total * 100 if unique_total else math.nan
		)

		tp_strict = share_within_tolerance
		tp_lenient = share_within_tolerance + matches_within_tolerance
		fp = detected_only
		fn = ground_truth_only

		precision_strict, recall_strict, f1_strict, accuracy_strict = _core_metrics(
			tp_strict, fp, fn
			)
		precision_lenient, recall_lenient, f1_lenient, accuracy_lenient = _core_metrics(
			tp_lenient, fp, fn
			)

		rows.append(
			{
					"recording_id": item.recording_id,
					"unique_total":unique_total,
					"total_detected": total_detected,
					"total_ground_truth": total_ground_truth,
					"ground_truth_only": ground_truth_only,
					"detected_only": detected_only,
					"share_within_tolerance": share_within_tolerance,
					"matches_within_tolerance": matches_within_tolerance,
					"pairs_outside_tolerance": pairs_outside_tolerance,
					"share_within_tolerance_percent": share_within_tolerance_percent,
					"tp_strict": tp_strict,
					"tp_lenient": tp_lenient,
					"fp": fp,
					"fn": fn,
					"precision_strict": precision_strict,
					"recall_strict": recall_strict,
					"f1_strict": f1_strict,
					"accuracy_strict": accuracy_strict,
					"precision_lenient": precision_lenient,
					"recall_lenient": recall_lenient,
					"f1_lenient": f1_lenient,
					"accuracy_lenient": accuracy_lenient,
					# "onset_mae_sec": item.onset_mae_sec,
					}
			)

	summary = pd.DataFrame(rows)
	if not summary.empty:
		summary = summary.sort_values("recording_id").reset_index(drop=True)
	return summary


def build_overall_summary(summary: pd.DataFrame) -> pd.Series:
	if summary.empty:
		return pd.Series(dtype=float)

	tp_strict_total = summary["tp_strict"].sum(skipna=True)
	tp_lenient_total = summary["tp_lenient"].sum(skipna=True)
	fp_total = summary["fp"].sum(skipna=True)
	fn_total = summary["fn"].sum(skipna=True)

	def _macro(column: str) -> float:
		return summary[column].mean(skipna=True)

	def _micro(tp: float) -> tuple[float, float, float, float]:
		precision = tp / (tp + fp_total) if (tp + fp_total) else math.nan
		recall = tp / (tp + fn_total) if (tp + fn_total) else math.nan
		if math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0:
			f1 = math.nan
		else:
			f1 = 2 * precision * recall / (precision + recall)
		accuracy = tp / (tp + fp_total + fn_total) if (tp + fp_total + fn_total) else math.nan
		return precision, recall, f1, accuracy

	precision_strict_micro, recall_strict_micro, f1_strict_micro, accuracy_strict_micro = _micro(
		tp_strict_total
		)
	(
			precision_lenient_micro,
			recall_lenient_micro,
			f1_lenient_micro,
			accuracy_lenient_micro,
			) = _micro(tp_lenient_total)

	# mae_values = summary["onset_mae_sec"].dropna()
	# mae_mean = mae_values.mean() if not mae_values.empty else math.nan

	return pd.Series(
		{
				"recording_count": len(summary),
				"total_detected_total": summary["total_detected"].sum(skipna=True),
				"total_ground_truth_total": summary["total_ground_truth"].sum(skipna=True),
				"share_within_tolerance_total": summary["share_within_tolerance"].sum(
					skipna=True
					),
				"matches_within_tolerance_total": summary["matches_within_tolerance"].sum(
					skipna=True
					),
				"pairs_outside_tolerance_total": summary[
					"pairs_outside_tolerance"
				].sum(skipna=True),
				"detected_only_total": summary["detected_only"].sum(skipna=True),
				"ground_truth_only_total": summary["ground_truth_only"].sum(skipna=True),
				"tp_strict_total": tp_strict_total,
				"tp_lenient_total": tp_lenient_total,
				"fp_total": fp_total,
				"fn_total": fn_total,
				"precision_strict_macro": _macro("precision_strict"),
				"recall_strict_macro": _macro("recall_strict"),
				"f1_strict_macro": _macro("f1_strict"),
				"accuracy_strict_macro": _macro("accuracy_strict"),
				"precision_lenient_macro": _macro("precision_lenient"),
				"recall_lenient_macro": _macro("recall_lenient"),
				"f1_lenient_macro": _macro("f1_lenient"),
				"accuracy_lenient_macro": _macro("accuracy_lenient"),
				"precision_strict_micro": precision_strict_micro,
				"recall_strict_micro": recall_strict_micro,
				"f1_strict_micro": f1_strict_micro,
				"accuracy_strict_micro": accuracy_strict_micro,
				"precision_lenient_micro": precision_lenient_micro,
				"recall_lenient_micro": recall_lenient_micro,
				"f1_lenient_micro": f1_lenient_micro,
				"accuracy_lenient_micro": accuracy_lenient_micro,
				# "onset_mae_sec_mean": mae_mean,
				}
		)
