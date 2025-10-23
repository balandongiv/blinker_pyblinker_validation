function init_eeglab_blinker(eeglabRoot)
%INIT_EEGLAB_BLINKER Initialize EEGLAB headlessly and ensure Blinker utilities are on path.

    % If not provided, use your known-good install
    if nargin < 1 || isempty(eeglabRoot)
        eeglabRoot = 'D:\code development\matlab_plugin\eeglab2025.1.0';
    end

    % Avoid path pollution from other installs
    if ~exist('eeglab', 'file')
        addpath(eeglabRoot);
    else
        % If eeglab is already on path, ensure it's the same root
        currentRoot = fileparts(which('eeglab.m'));
        if ~strcmpi(currentRoot, eeglabRoot)
            warning('Switching EEGLAB root from:\n  %s\nto:\n  %s', currentRoot, eeglabRoot);
            rmpath(genpath(currentRoot));
            addpath(eeglabRoot);
        end
    end

    % Start EEGLAB silently (no GUI)
    eeglab('nogui');

    % Ensure Blinker plugin (including utilities) is on path
    blinkerDir = fullfile(eeglabRoot, 'plugins', 'Blinker1.2.0');
    if exist(blinkerDir, 'dir')
        addpath(genpath(blinkerDir));
    else
        warning('Blinker plugin not found at %s', blinkerDir);
    end

    % Optional sanity checks
    needed = {'getBlinkerDefaults', 'extractBlinksEEG', 'extractBlinkProperties'};
    for k = 1:numel(needed)
        if ~exist(needed{k}, 'file')
            warning('Function not found on path: %s', needed{k});
        end
    end
end
