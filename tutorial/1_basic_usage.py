'''
Implement the blink detect as per the legacy MATLAB-based BLINK approach, but using Python and MNE-Python.
This code is a basic usage example of the pyblinker library for detecting eye blinks in EEG data.
It reads a raw EEG file, processes the data to filter and resample it,
and then uses the BlinkDetector class to identify and visualize eye blinks.

'''

import logging
import os


import matplotlib
import mne
from pyblinker.blinker.pyblinker import BlinkDetector
logging.basicConfig(level=logging.INFO)

matplotlib.use('TkAgg')

def plot_blinks(raw_file):
    """Plot eye close events based on EEG signals.

    Args:
    raw_file (str): Path to the raw EEG candidate_signal file in .fif format.

    Returns:
    None
    """
    raw = mne.io.read_raw_fif(raw_file, preload=True)
    raw.pick_types(eeg=True)
    raw.filter(0.5, 20.5, fir_design='firwin')
    raw.resample(100)


    drange=[f'EEG 00{X}' for X in range (10)]
    to_drop_ch = list(set(raw.ch_names) - set(drange))
    raw = raw.drop_channels(to_drop_ch)

    annot, ch, number_good_blinks, blink_details, fig_data, ch_selected = BlinkDetector(raw, visualize=False, annot_label=None,
                                                                             filter_low=0.5, filter_high=30.0, resample_rate=100,
                                                                             n_jobs=2,use_multiprocessing=True).get_blink()
    raw.set_annotations(annot)
    raw.plot(block=True, title=f'Eye close based on channel {ch}')

if __name__ == '__main__':

    sample_data_folder = mne.datasets.sample.data_path()
    sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample', 'sample_audvis_filt-0-40_raw.fif')
    plot_blinks(sample_data_raw_file)
