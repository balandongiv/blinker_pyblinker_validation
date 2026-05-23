function out = run_blinker_pipeline_wrap_set(setFile, paramsIn)
    if nargin < 2
        [blinks, blinkFits, blinkProps, blinkStats, params, com] = run_blinker_pipeline_set(setFile);
    else
        [blinks, blinkFits, blinkProps, blinkStats, params, com] = run_blinker_pipeline_set(setFile, paramsIn);
    end

    out = struct( ...
        'blinks',     blinks, ...
        'blinkFits',  blinkFits, ...
        'blinkProps', blinkProps, ...
        'blinkStats', blinkStats, ...
        'params',     params, ...
        'com',        com);

    out = make_engine_safe(out);
end

function y = make_engine_safe(x)
    if istable(x)
        x = table2struct(x);
    end

    if isstruct(x)
        if ~isscalar(x)
            y = arrayfun(@(s) make_engine_safe(s), x, 'UniformOutput', false);
            return
        end
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

    if isstring(x),       y = cellstr(x);          return; end
    if isa(x,'datetime'), y = cellstr(string(x));   return; end

    if isobject(x) && ~isnumeric(x) && ~islogical(x) && ~ischar(x)
        try
            y = make_engine_safe(struct(x));
        catch
            try
                y = char(x);
            catch
                y = [];
            end
        end
        return
    end

    y = x;
end
