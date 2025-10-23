function out = run_blinker_pipeline_wrap(edfFile)
    % Run your existing pipeline
    [blinks, blinkFits, blinkProps, blinkStats, params, com] = run_blinker_pipeline(edfFile);

    % (Optional) if any are non-scalar structs and you prefer cells:
    % if ~isscalar(blinkFits),  blinkFits  = num2cell(blinkFits);  end
    % if ~isscalar(blinkProps), blinkProps = num2cell(blinkProps); end

    % Return as a CELL (top-level is not a struct => engine is happy)
    out = {blinks, blinkFits, blinkProps, blinkStats, params, com};
end
