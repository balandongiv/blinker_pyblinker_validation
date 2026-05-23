function [blinks, blinkFits, blinkProperties, blinkStatistics, params, com] = run_blinker_pipeline_set(setPath, paramsIn)
% RUN_BLINKER_PIPELINE_SET  Load EEGLAB .set file, run BLINKER pipeline.
%
% Identical to run_blinker_pipeline except it uses pop_loadset instead of
% pop_biosig so it handles EEGLAB .set/.fdt recordings (e.g. Cao dataset).
%
% USAGE:
%   [blinks, blinkFits, blinkProps, blinkStats, params, com] = ...
%       run_blinker_pipeline_set('C:\path\to\data.set');

    %% --- Load EEGLAB .set file
    [fdir, fname, ~] = fileparts(setPath);
    EEG = pop_loadset('filename', [fname '.set'], 'filepath', fdir);

    %% --- Build/normalize params (non-GUI path)
    if nargin < 2 || isempty(paramsIn)
        params = getBlinkerDefaults(EEG);
    else
        params = paramsIn;
    end
    [params, okay] = normalizeParams(params); %#ok<ASGLU>

    %% Check the defaults to make sure all is there
    [params, errors] = checkBlinkerDefaults(params, getBlinkerDefaults(EEG));
    if ~isempty(errors)
        error('pop_blinker:BadParameters', ['|' sprintf('%s|', errors{:})]);
    end

    %% Extract the candidate signals
    logIfVerbose(params, 'Extracting blinks.....\n');
    [blinks, params] = extractBlinksEEG(EEG, params);

    %% --- Finalize blinks structure metadata
    blinks.fileName   = safeFileName(EEG, params);
    blinks.experiment = params.experiment;
    blinks.subjectID  = params.subjectID;
    blinks.uniqueName = params.uniqueName;
    blinks.task       = params.task;

    startTime = '';
    try
        startTime = sprintf('%s %s', params.startDate, params.startTime);
        blinks.startTime = parseStartTime(startTime);
    catch ME
        warning('pop_blinker:BadStartTime', ...
            '%s has invalid start time (%s), using NaN [%s]\n', ...
            blinks.fileName, startTime, ME.message);
        blinks.startTime = NaN;
    end

    if isempty(blinks.status)
        blinks.status = 'success';
    end

    if isempty(blinks.usedSignal) || isnan(blinks.usedSignal)
        warning('pop_blinker:NoBlinks', '%s does not have blinks\n', blinks.fileName);
        blinkFits        = [];
        blinkProperties  = [];
        blinkStatistics  = [];
        com = sprintf('pop_blinker(%s, %s);', 'EEG', struct2str(params));
        return;
    end

    signalNumbers = cellfun(@double, {blinks.signalData.signalNumber});
    signalIndex   = find(signalNumbers == abs(blinks.usedSignal), 1, 'first');
    signalData    = blinks.signalData(signalIndex);

    %% --- Per-blink properties & stats
    logIfVerbose(params, 'Extracting blink shapes and properties.....\n');
    [blinkProperties, blinkFits] = extractBlinkProperties(signalData, params);

    logIfVerbose(params, 'Extracting blink statistics.....\n');
    blinkStatistics = extractBlinkStatistics(blinks, blinkFits, blinkProperties, params);

    if params.verbose
        outputBlinkStatistics(blinkStatistics);
    end

    safeCall(@() saveBlinkerStructures(params, blinks, blinkFits, blinkProperties, blinkStatistics), ...
             'pop_blinker:SaveStructureError', params);
    safeCall(@() maybeShowMaxDistribution(params, blinks, blinkFits), ...
             'pop_blinker:SaveDistributionError', params);
    safeCall(@() maybeDumpBlinkImages(params, blinks, signalIndex), ...
             'pop_blinks:SaveDumpImagesError', params);
    safeCall(@() maybeDumpBlinkPositions(params, blinks, blinkFits), ...
             'pop_blinks:SaveDumpPositionsError', params);

    com = sprintf('pop_blinker(%s, %s);', inputname(1), struct2str(params));
end

% ============================== Helpers ===================================

function t = parseStartTime(startTimeStr)
    try
        dt = datetime(startTimeStr, 'InputFormat', 'yyyy-MM-dd HH:mm:ss', 'TimeZone', 'local');
    catch
        try
            dt = datetime(startTimeStr, 'InputFormat', 'yyyy/MM/dd HH:mm:ss', 'TimeZone', 'local');
        catch
            dt = datetime(startTimeStr, 'TimeZone', 'local');
        end
    end
    if isnat(dt)
        t = NaN;
    else
        t = datenum(dt);
    end
end

function fn = safeFileName(EEG, params)
    if isfield(params, 'fileName') && ~isempty(params.fileName)
        fn = params.fileName;
        return;
    end
    pth = '';
    if isfield(EEG, 'filepath') && ~isempty(EEG.filepath)
        pth = EEG.filepath;
    end
    fname = '';
    if isfield(EEG, 'filename') && ~isempty(EEG.filename)
        fname = EEG.filename;
    end
    if ~isempty(pth) && ~isempty(fname)
        fn = fullfile(pth, fname);
    else
        fn = fname;
    end
end

function logIfVerbose(params, fmt, varargin)
    if isfield(params, 'verbose') && params.verbose
        fprintf(fmt, varargin{:});
    end
end

function safeCall(fcn, errTag, params)
    try
        fcn();
    catch ME
        fprintf('%s %s\n', errTag, ME.message);
        if isfield(params, 'verbose') && params.verbose
            for k = 1:numel(ME.stack)
                s = ME.stack(k);
                fprintf('  at %s (%s:%d)\n', s.name, s.file, s.line);
            end
        end
    end
end

function saveBlinkerStructures(params, blinks, blinkFits, blinkProperties, blinkStatistics)
    if ~isfield(params,'dumpBlinkerStructures') || ~params.dumpBlinkerStructures
        return;
    end
    logIfVerbose(params, 'Saving the blink structures ......\n');
    thePath = fileparts(params.blinkerSaveFile);
    if ~isempty(thePath) && ~exist(thePath, 'dir')
        mkdir(thePath);
    end
    save(params.blinkerSaveFile, 'blinks', 'blinkFits', 'blinkProperties', 'blinkStatistics', 'params', '-v7.3');
end

function maybeShowMaxDistribution(params, blinks, blinkFits)
    if isfield(params,'showMaxDistribution') && params.showMaxDistribution
        showMaxDistribution(blinks, blinkFits);
    end
end

function maybeDumpBlinkImages(params, blinks, signalIndex)
    if ~isfield(params,'dumpBlinkImages') || ~params.dumpBlinkImages
        return;
    end
    traceName  = blinks.signalData(signalIndex).signalLabel;
    blinkTrace = blinks.signalData(signalIndex).signal;
    dumpIndividualBlinks(blinks, blinkFits, blinkTrace, traceName, params.blinkerDumpDir, params.correlationThresholdTop);
end

function maybeDumpBlinkPositions(params, blinks, blinkFits)
    if ~isfield(params,'dumpBlinkPositions') || ~params.dumpBlinkPositions
        return;
    end
    fileOut = fullfile(params.blinkerDumpDir, [blinks.uniqueName 'leftZeros.txt']);
    dumpDatasetLeftZeros(fileOut, blinkFits, blinks.srate);
end
