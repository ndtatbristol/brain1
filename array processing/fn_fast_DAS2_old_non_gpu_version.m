function [result, varargout] = fn_fast_DAS2(exp_data, focal_law)
%USAGE
%   [result, [timings]] = fn_fast_DAS2(exp_data, focal_law)
%SUMMARY
%   Delay and sum implementation that can be used with any linear imaging
%   algorithm where tx and rx weights are frequency independent. Can be
%   used either with same focal law on transmit and receive (in which case
%   focal_law.lookup_time, focal_law.lookup_ind and focal_law.lookup_amp
%   must exist) or with separate transmit and receive focal laws (in which 
%   case focal_law.lookup_time_tx, focal_law.lookup_time_rx etc. must
%   exist).
%INPUTS
%   exp_data - experimental data in usual format
%   focal_law - structured variable that effectively contains imaging
%   algorithm (described below for same tx and rx focal law - if different
%   laws are required then two sets of following are needed suffixed with
%   _tx and _rx).
%   focal_law.lookup_time - 3D matrix of propagation times from each element
%   to each image point
%   focal_law.lookup_ind - 3D matrix of equivalent indices (round(t/dt)) from
%   each element to each image point
%   focal_law.lookup_amp - 3D matrix of weightings associated with each
%   element and each image point
%   focal_law.hilbert_on - flag to do hilbert transform of each signal prior
%   to processing
%   focal_law.filter_on - flag to include pre-filtering of time-traces
%   focal_law.filter - frequency domain filter to apply to time-traces if
%   required
%   focal_law.tt_ind - optional indices of each time-trace to process (if not
%   specified, they will all be processed, regardless of whether or not
%   they make any contributions to image, so using this vector can save
%   time with algorithms such as b-scans where certain time-traces in FMC
%   data are not used at all)
%   focal_law.tt_weight - weighting associated with each time trace (used to
%   double pitch-catch signals if half matrix capture used)
%   focal_law.interpolation_method = sets how time-domain data is
%   interpolated; 'nearest' means no interpolation and nearest point is
%   used; 'linear' means linear interpolation
%OUTPUTS
%   result - the resulting image
%   timings - if specified returns times taken for different steps of
%   calculation for debugging/comparison purposes.

%--------------------------------------------------------------------------

tic
if nargout > 1
    record_times = 1;
else
    record_times = 0;
end

if isfield(focal_law, 'lookup_time')
    result_dims = size(focal_law.lookup_time);
else
    if prod(size(focal_law.lookup_time_tx)) > prod(size(focal_law.lookup_time_rx))
        result_dims = size(focal_law.lookup_time_tx);
    else
        result_dims = size(focal_law.lookup_time_rx);
    end
end
result_dims = result_dims(1:end - 1);
result = zeros(result_dims);
result = result(:);%vectorise

if ~isfield(focal_law, 'tt_ind')
    focal_law.tt_ind = 1:length(exp_data.tx);
end

if isfield(focal_law, 'lookup_time')
    sep_tx_rx_laws = 0;
    focal_law.lookup_time = reshape(focal_law.lookup_time, prod(result_dims), []);
    focal_law.lookup_ind = reshape(focal_law.lookup_ind, prod(result_dims), []);
    focal_law.lookup_amp = reshape(focal_law.lookup_amp, prod(result_dims), []);
else
    sep_tx_rx_laws = 1;
    focal_law.lookup_time_tx = reshape(focal_law.lookup_time_tx, prod(result_dims), []);
    focal_law.lookup_ind_tx = reshape(focal_law.lookup_ind_tx, prod(result_dims), []);
    focal_law.lookup_amp_tx = reshape(focal_law.lookup_amp_tx, prod(result_dims), []);
    focal_law.lookup_time_rx = reshape(focal_law.lookup_time_rx, prod(result_dims), []);
    focal_law.lookup_ind_rx = reshape(focal_law.lookup_ind_rx, prod(result_dims), []);
    focal_law.lookup_amp_rx = reshape(focal_law.lookup_amp_rx, prod(result_dims), []);
end;

% if ~focal_law.amp_on
%     if sep_tx_rx_laws
%         focal_law.lookup_amp_tx = ones(size(focal_law.lookup_amp_tx));
%         focal_law.lookup_amp_rx = ones(size(focal_law.lookup_amp_rx));
%     else
%         focal_law.lookup_amp = ones(size(focal_law.lookup_amp));
%     end
% end

%see if data is HMC unless explicitly stated by looking for and tt_weight
%values of 2
if ~isfield(focal_law, 'hmc_data')
    hmc_data = any(focal_law.tt_weight == 2); %HMC needs to be considered differently if sep tx and rx laws are used
else
    hmc_data = focal_law.hmc_data;
end

%filtering
if focal_law.filter_on
    exp_data.time_data = ifft(spdiags(focal_law.filter, 0, length(exp_data.time), length(exp_data.time)) * fft(exp_data.time_data));
else
    if focal_law.hilbert_on
        exp_data.time_data = ifft(spdiags([1:length(exp_data.time)]' < length(exp_data.time) / 2, 0, length(exp_data.time), length(exp_data.time)) * fft(exp_data.time_data));
    end
end

%add line of zeros to tail of data so that out of range values are zero
%in interpolation or lookup stage

exp_data.time_data = [exp_data.time_data; zeros(1, size(exp_data.time_data, 2))];
n = size(exp_data.time_data, 1);

if record_times
    varargout{1}.filter = double(toc);
end

if sep_tx_rx_laws
    if hmc_data
        method = 'Different Tx and RX laws, HMC data';
    else
        method = 'Different Tx and RX laws, FMC data';
    end
else
    method = 'Same Tx and RX laws';
end

method = [method, ' (', focal_law.interpolation_method, ')'];

%Now the main loops - different loop depending on method.
switch method
    case 'Same Tx and RX laws (nearest)'
        
        for ii = 1:length(focal_law.tt_ind)
            jj = focal_law.tt_ind(ii);
            amp1 = ...
                focal_law.lookup_amp(:, exp_data.tx(jj)) .* ...
                focal_law.lookup_amp(:, exp_data.rx(jj)) * ...
                focal_law.tt_weight(jj);
            ind1 = ...
                focal_law.lookup_ind(:, exp_data.tx(jj)) + ...
                focal_law.lookup_ind(:, exp_data.rx(jj));
            %out of range indices set to n (to give a zero value in result)
            ind1(find(ind1 < 1)) = n;
            ind1(find(ind1 > n)) = n;
            result = result + exp_data.time_data(ind1, jj) .* amp1(:);
        end
        
    case 'Different Tx and RX laws, FMC data (nearest)'
        %Separate focal laws for Tx and Rx and FMC data
        for ii = 1:length(focal_law.tt_ind)
            jj = focal_law.tt_ind(ii);
            amp1 = ...
                focal_law.lookup_amp_tx(:, exp_data.tx(jj)) .* ...
                focal_law.lookup_amp_rx(:, exp_data.rx(jj)) * ...
                focal_law.tt_weight(jj);
            ind1 = ...
                focal_law.lookup_ind_tx(:, exp_data.tx(jj)) + ...
                focal_law.lookup_ind_rx(:, exp_data.rx(jj));
            %out of range indices set to n (to give a zero value in result)
            ind1(find(ind1 < 1)) = n;
            ind1(find(ind1 > n)) = n;
            result = result + exp_data.time_data(ind1, jj) .* amp1(:);
        end
        
    case 'Different Tx and RX laws, HMC data (nearest)'

        for ii = 1:length(focal_law.tt_ind)
            jj = focal_law.tt_ind(ii);
            ind1 = ...
                focal_law.lookup_ind_tx(:, exp_data.tx(jj)) + ...
                focal_law.lookup_ind_rx(:, exp_data.rx(jj));
            amp1 = ...
                focal_law.lookup_amp_tx(:, exp_data.tx(jj)) .* ...
                focal_law.lookup_amp_rx(:, exp_data.rx(jj)) * ...
                focal_law.tt_weight(jj) / 2;
            ind2 = ...
                focal_law.lookup_ind_tx(:, exp_data.rx(jj)) + ...
                focal_law.lookup_ind_rx(:, exp_data.tx(jj));
            amp2 = ...
                focal_law.lookup_amp_tx(:, exp_data.rx(jj)) .* ...
                focal_law.lookup_amp_rx(:, exp_data.tx(jj)) * ...
                focal_law.tt_weight(jj) / 2;
            %out of range indices set to n (to give a zero value in result)
            ind1(find(ind1 < 1)) = n;
            ind1(find(ind1 > n)) = n;
            ind2(find(ind2 < 1)) = n;
            ind2(find(ind2 > n)) = n;
            result = result + ...
                exp_data.time_data(ind1, jj) .* amp1 + ...
                exp_data.time_data(ind2, jj) .* amp2;
        end
        
%Linear interpolation methods
        
    case 'Same Tx and RX laws (linear)'
        
        for ii = 1:length(focal_law.tt_ind)
            jj = focal_law.tt_ind(ii);
            amp1 = ...
                focal_law.lookup_amp(:, exp_data.tx(jj)) .* ...
                focal_law.lookup_amp(:, exp_data.rx(jj)) * ...
                focal_law.tt_weight(jj);
            time1 = ...
                focal_law.lookup_time(:, exp_data.tx(jj)) + ...
                focal_law.lookup_time(:, exp_data.rx(jj));
            result = result + ...
                fn_fast_linear_interp(exp_data.time, exp_data.time_data(:, jj), time1(:)) .* amp1(:);
        end
        
    case 'Different Tx and RX laws, FMC data (linear)'
        %Separate focal laws for Tx and Rx and FMC data
        for ii = 1:length(focal_law.tt_ind)
            jj = focal_law.tt_ind(ii);
            amp1 = ...
                focal_law.lookup_amp_tx(:, exp_data.tx(jj)) .* ...
                focal_law.lookup_amp_rx(:, exp_data.rx(jj)) * ...
                focal_law.tt_weight(jj);
            time1 = ...
                focal_law.lookup_time_tx(:, exp_data.tx(jj)) + ...
                focal_law.lookup_time_rx(:, exp_data.rx(jj));
            result = result + ...
                fn_fast_linear_interp(exp_data.time, exp_data.time_data(:, jj), time1(:)) .* amp1(:);
        end
        
    case 'Different Tx and RX laws, HMC data (linear)'

        for ii = 1:length(focal_law.tt_ind)
            jj = focal_law.tt_ind(ii);
            time1 = ...
                focal_law.lookup_time_tx(:, exp_data.tx(jj)) + ...
                focal_law.lookup_time_rx(:, exp_data.rx(jj));
            amp1 = ...
                focal_law.lookup_amp_tx(:, exp_data.tx(jj)) .* ...
                focal_law.lookup_amp_rx(:, exp_data.rx(jj)) * ...
                focal_law.tt_weight(jj) / 2;
            time2 = ...
                focal_law.lookup_time_tx(:, exp_data.rx(jj)) + ...
                focal_law.lookup_time_rx(:, exp_data.tx(jj));
            amp2 = ...
                focal_law.lookup_amp_tx(:, exp_data.rx(jj)) .* ...
                focal_law.lookup_amp_rx(:, exp_data.tx(jj)) * ...
                focal_law.tt_weight(jj) / 2;
            result = result + ...
                fn_fast_linear_interp(exp_data.time, exp_data.time_data(:, jj), time1(:)) .* amp1(:) + ...
                fn_fast_linear_interp(exp_data.time, exp_data.time_data(:, jj), time2(:)) .* amp2(:);
        end
        
end

%finally reshape image back to desired dimensions
result = reshape(result, result_dims);%back to right shape

if record_times
    varargout{1}.loop = double(toc) - varargout{1}.filter;
    varargout{1}.total = double(toc);
end
end

function yi = fn_fast_linear_interp(x, y, xi)
%note out of range xi values return zero
x = x(:);
y = y(:);
xi = xi(:);
dx = x(2) - x(1);
x0 = x(1);
ii = floor((xi - x0) / dx) + 1;
j1 = find(ii < 1);
ii(j1) = 1;
j2 = find(ii > (length(x) - 1));
ii(j2) = length(x) - 1;
yi = y(ii) + (y(ii + 1) - y(ii)) .* (xi - x(ii)) ./ dx;
yi(j1) = 0;
yi(j2) = 0;
end