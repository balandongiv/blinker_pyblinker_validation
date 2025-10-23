init_eeglab_blinker('D:\code development\matlab_plugin\eeglab2025.1.0');

edfFile = 'C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\src\matlab_runner\seg_annotated_raw.edf';
[blinks, blinkFits, blinkProps, blinkStats, params, com] = run_blinker_pipeline(edfFile);
