function out = run_blinker_pipeline_wrap(edfFile)
    % Call the original function
    [blinks, blinkFits, blinkProps, blinkStats, params, com] = run_blinker_pipeline(edfFile);

    % Pack into a single scalar struct
    out = struct( ...
        'blinks',     blinks, ...
        'blinkFits',  blinkFits, ...
        'blinkProps', blinkProps, ...
        'blinkStats', blinkStats, ...
        'params',     params, ...
        'com',        com);

    % Make everything safe for the MATLAB Engine (no non-scalar structs inside)
    out = make_engine_safe(out);
end

function y = make_engine_safe(x)
    % Tables -> struct array (then handled below)
    if istable(x)
        x = table2struct(x);  % 1xN struct
    end

    if isstruct(x)
        % If it's a struct array, convert to a cell array of scalar structs
        if ~isscalar(x)
            y = arrayfun(@(s) make_engine_safe(s), x, 'UniformOutput', false);
            return
        end
        % Recurse into fields
        f = fieldnames(x);
        for k = 1:numel(f)
            x.(f{k}) = make_engine_safe(x.(f{k}));
        end
        y = x;
        return
    end

    if iscell(x)
        y = cellfun(@make_engine_safe, x, 'UniformOutput', false);
        return
    end

    % Normalize some common object types
    if isstring(x),     y = cellstr(x);       return; end
    if isa(x,'datetime'), y = cellstr(string(x)); return; end

    % Fallback for other MATLAB objects: try struct(), else char()
    if isobject(x) && ~isnumeric(x) && ~islogical(x) && ~ischar(x)
        try
            y = make_engine_safe(struct(x));
        catch
            try
                y = char(x);
            catch
                y = []; % last resort: drop unsupported objects
            end
        end
        return
    end

    % Plain numerics/logicals/chars pass through
    y = x;
end
