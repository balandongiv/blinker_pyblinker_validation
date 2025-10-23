function out = run_blinker_pipeline_wrap(edfFile)
    % Run your existing pipeline
    [blinks, blinkFits, blinkProps, blinkStats, params, com] = run_blinker_pipeline(edfFile);

    % The MATLAB Engine for Python can only marshal *scalar* structs.  The
    % Blinker pipeline returns several struct arrays, so we convert the
    % problematic ones into cell arrays of scalar structs before handing them
    % back to Python.
    blinkFits  = ensure_scalar_structs(blinkFits);
    blinkProps = ensure_scalar_structs(blinkProps);
    blinkStats = ensure_scalar_structs(blinkStats);
    params     = ensure_scalar_structs(params);

    % Package everything into a single scalar struct so Python gets an object
    % that behaves like a dictionary (``out['field']``).
    out = struct();
    out.blinks = blinks;
    out.blinkFits = blinkFits;
    out.blinkProps = blinkProps;
    out.blinkStats = blinkStats;
    out.params = params;
    out.com = com;
end

function value = ensure_scalar_structs(value)
    if isstruct(value) && ~isscalar(value)
        % ``arrayfun`` with ``'UniformOutput', false`` turns an array of structs
        % into a cell array where each element is a scalar struct.  The MATLAB
        % Engine is happy to return these cells to Python.
        value = arrayfun(@(s) s, value, 'UniformOutput', false);
    end
end
