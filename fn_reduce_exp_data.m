function exp_data = fn_reduce_exp_data(orig_exp_data, min_time, max_time, down_sample)
%SUMMARY
%   Reduces experimental data structure to (a) time range between specified
%   limits, (b) HMC data (by averaging) if it was FMC originally and
%   (c) downsampled data
%INPUTS
%   orig_exp_data - data to be reduced
%   min_time, max_time - time window to retain
%   down_sample - factor to down sample by (1 = no downsampling)
%OUTPUTS
%   exp_data - the reduced data

%--------------------------------------------------------------------------
exp_data = orig_exp_data;

%ensure down_sample is integer >= 1;
down_sample = round(max([1, down_sample]));

%reduce time range
[dummy, i1] = min(abs(min_time - exp_data.time));
[dummy, i2] = min(abs(max_time - exp_data.time));
exp_data.time = exp_data.time(i1:down_sample:i2);
exp_data.time_data = exp_data.time_data(i1:down_sample:i2, :);

%reduce if FMC data to HMC
ii = 1;
cols_to_keep = ones(size(exp_data.tx));
for ii = 1:length(exp_data.tx)
    if ~cols_to_keep(ii)
        continue
    end
    %pulse-echo data needs no action
    if exp_data.tx(ii) == exp_data.rx(ii)
        continue
    end
    %look for reciprocal combo
    jj = find(exp_data.tx == exp_data.rx(ii) & exp_data.rx == exp_data.tx(ii) & cols_to_keep);
    if ~isempty(jj)
        %make current column equal to mean of itself and reciprocal version
        exp_data.time_data(:, ii) = mean(exp_data.time_data(:, [ii, jj]), 2);
        %delete reciprocal column
        cols_to_keep(jj) = 0;
    end
end
jj = find(cols_to_keep);
exp_data.time_data = exp_data.time_data(:, jj);
exp_data.tx = exp_data.tx(:, jj);
exp_data.rx = exp_data.rx(:, jj);
end