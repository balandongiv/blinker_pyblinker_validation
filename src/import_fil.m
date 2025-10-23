

% --- Load EDF (your code)
edfPath = 'C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\src\matlab_runner\seg_annotated_raw.edf';
[EEG, ~, hdr] = pop_biosig(edfPath);   %  % hdr = 'dat' from pop_biosig
[params] = getBlinkerDefaults(EEG)

[params, okay] = normalizeParams(params)

 %% Check the defaults to make sure all is there
 [params, errors] = checkBlinkerDefaults(params, getBlinkerDefaults(EEG));


%%%%%%%%%%%

 %% Check the defaults to make sure all is there
     [params, errors] = checkBlinkerDefaults(params, getBlinkerDefaults(EEG));
      if ~isempty(errors)
          error('pop_blinker:BadParameters', ['|' sprintf('%s|', errors{:})]);
      end
     %% Extract the blinks
     if params.verbose
         fprintf('Extracting blinks.....\n');
     end
     [blinks, params] = extractBlinksEEG(EEG, params);
     
     %% Finalize the blinks structure
     blinks.fileName = [EEG.filepath EEG.filename];
     blinks.experiment = params.experiment;
     blinks.subjectID = params.subjectID;
     blinks.uniqueName = params.uniqueName;
     blinks.task = params.task;
     blinks.fileName = params.fileName;
     startTime = '';
     try
         startTime = [params.startDate ' ' params.startTime];
         blinks.startTime = datenum(startTime, -1);
     catch Mex
         warning('pop_blinker:BadStartTime', ...
             '%s has invalid start time (%s), using NaN [%s]\n', ...
             blinks.fileName, startTime, Mex.message);
     end

     if isempty(blinks.status)
         blinks.status = 'success';
     end
     if isempty(blinks.usedSignal) || isnan(blinks.usedSignal)
         warning('pop_blinker:NoBlinks', ...
                '%s does not have blinks\n', blinks.fileName);
         return;
     end
     signalNumbers = cellfun(@double, {blinks.signalData.signalNumber});
     signalIndex = find(signalNumbers == abs(blinks.usedSignal), 1, 'first');
     signalData = blinks.signalData(signalIndex);
     
     %% Calculate the blink properties
     if params.verbose
         fprintf('Extracting blink shapes and properties.....\n');
     end
     [blinkProperties, blinkFits] = extractBlinkProperties(signalData, params);
     if params.verbose
         fprintf('Extracting blink statistics.....\n');
     end
     blinkStatistics = extractBlinkStatistics(blinks, blinkFits, ...
                                              blinkProperties, params);
     %% Calculate summary statistics
     if params.verbose
         outputBlinkStatistics(blinkStatistics);
     end
     
     %% Saving the structures
     try
         if params.dumpBlinkerStructures
             if params.verbose
                fprintf('Saving the blink structures ......\n');
             end
             thePath = fileparts(params.blinkerSaveFile);
             if ~exist(thePath, 'dir')
                 mkdir(thePath);
             end
             save(params.blinkerSaveFile, 'blinks', 'blinkFits', ...
                 'blinkProperties', 'blinkStatistics', 'params', '-v7.3');
         end
     catch Mex
         fprintf('pop_blinker:SaveStructureError %s\n', Mex.message');
     end
     try
         if params.showMaxDistribution
             showMaxDistribution(blinks, blinkFits);
         end
     catch Mex
         fprintf('pop_blinker:SaveDistributionError %s\n', Mex.message');
     end
     try
         if params.dumpBlinkImages
             traceName = blinks.signalData(signalIndex).signalLabel;
             blinkTrace = blinks.signalData(signalIndex).signal;
             dumpIndividualBlinks(blinks, blinkFits, blinkTrace, ...
                 traceName, params.blinkerDumpDir, params.correlationThresholdTop);
         end
     catch Mex
         fprintf('pop_blinks:SaveDumpImagesError %s\n', Mex.message');
     end
     try
         if params.dumpBlinkPositions
             fileOut = [params.blinkerDumpDir filesep blinks.uniqueName 'leftZeros.txt'];
             dumpDatasetLeftZeros(fileOut, blinkFits, blinks.srate)
         end
     catch Mex
         fprintf('pop_blinks:SaveDumpImagesError %s\n', Mex.message');
     end
     com = sprintf('pop_blinker(%s, %s);', inputname(1), struct2str(params));