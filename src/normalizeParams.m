
function [o, ok] = normalizeParams(p)
        % Build outStruct exactly like the GUI would, but from 'params'
        ok = true;
        o = struct();

        % Simple copies (take .value)
        f = {'blinkerSaveFile','blinkerDumpDir','experiment','subjectID', ...
             'task','uniqueName','startDate','startTime'};
        for i = 1:numel(f)
            if isfield(p, f{i}) && isfield(p.(f{i}), 'value')
                o.(f{i}) = p.(f{i}).value;
            end
        end

        % Signal type (accept string or numeric)
        menu = {'UseNumbers','UseLabels','UseICs'};
        sti = p.signalTypeIndicator.value;
        if isnumeric(sti) && sti>=1 && sti<=numel(menu)
            o.signalTypeIndicator = menu{sti};
        elseif ischar(sti) || isstring(sti)
            ii = find(strcmpi(menu, char(sti)), 1, 'first');
            if ~isempty(ii), sti = menu{ii}; end
            o.signalTypeIndicator = char(sti);
        else
            o.signalTypeIndicator = 'UseNumbers';
        end

        % Booleans
        o.dumpBlinkerStructures = logical(p.dumpBlinkerStructures.value);
        o.showMaxDistribution   = logical(p.showMaxDistribution.value);
        o.dumpBlinkPositions    = logical(p.dumpBlinkPositions.value);
        o.dumpBlinkImages       = logical(p.dumpBlinkImages.value);

        % Signal numbers (accept numeric vector OR string)
        sn = [];
        if isfield(p,'signalNumbers') && isfield(p.signalNumbers,'value')
            v = p.signalNumbers.value;
            if isnumeric(v)
                sn = sort(v(:))';
            elseif ischar(v) || isstring(v)
                vv = str2num(char(v)); %#ok<ST2NM>
                if ~isempty(vv), sn = sort(vv(:))'; end
            end
        end
        if isempty(sn)
            sn = NaN;
        end
        o.signalNumbers = sn;

        % Signal labels (accept cellstr OR comma string)
        sl = {};
        if isfield(p,'signalLabels') && isfield(p.signalLabels,'value')
            v = p.signalLabels.value;
            if iscellstr(v)
                sl = sort(v);
            elseif ischar(v) || isstring(v)
                sl = str2cellstr(char(v));
            end
        end
        o.signalLabels = sl;
    end