function tt_ind = fn_optimise_focal_law2(focal_law, tx, rx)
%SUMMARY
%   This returns tt_ind vector of time-traces to actual process
%   in fn_fast_DAS2 by analysing which time-traces make contributions to
%   image.


tt_used = zeros(size(tx));

if isfield(focal_law, 'lookup_amp')
    result_dims = size(focal_law.lookup_amp);
else
    result_dims = size(focal_law.lookup_amp_tx);
end

result_dims = result_dims(1:end - 1);

if isfield(focal_law, 'lookup_amp')
    lookup_amp = reshape(focal_law.lookup_amp, prod(result_dims), []);
    for ii = 1:length(tx)
        tt_used(ii) = any(lookup_amp(:, tx(ii)) & lookup_amp(:,rx(ii)));
    end;
else
    lookup_amp_tx = reshape(focal_law.lookup_amp_tx, prod(result_dims), []);
    lookup_amp_rx = reshape(focal_law.lookup_amp_rx, prod(result_dims), []);
    for ii = 1:length(tx)
        tt_used(ii) = any(lookup_amp_tx(:, tx(ii)) & lookup_amp_rx(:,rx(ii)));
    end;
end

tt_ind = find(tt_used);

end