function [min_freq, cent_freq, max_freq, min_index, cent_index, max_index] = fn_calculate_bandwidth_of_spectrum(freq, spec, varargin);
%USAGE
%	[min_freq, cent_freq, max_freq, min_index, cent_index, max_index] = fn_calculate_bandwidth_of_spectrum(freq, spec [, db_down]);
%SUMMARY
%	Calculates the bandwidth of spectrum or spectra
%AUTHOR
%	Paul Wilcox (2003);
%INPUTS
%	freq - vector representing frequency axis for spectrum(s) or scalar
%	representing frequency step
%	spec - vector containing a single spectrum, or matrix containigng
%	a spectrum in each column, in which case, no. of rows must equal
%	length of freq vector. First element (row in matrix) is dc component of
%	spectrum
%	db_down[40] - number of dB down from peak amplitude on which bandwidth is
%	based
%OUTPUTS
%	min_freq, cent_freq and max_freq - the minimum, maximum and peak
%	frequencies. If signal is matrix, then these will be vectors with as
%	many elements as there are columns in signal.
%	min_index, cent_index, max_index - indices of above

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
default_db_down = 40;
if nargin < 3
	db_down = default_db_down;
else
	db_down = varargin{1};
end;
signal = squeeze(spec);
if length(freq) == 1
    freq_step = freq;
else
    freq_step = abs(freq(1) - freq(2));
end;

min_value = 10 ^ (-db_down / 20);

no_signals = size(spec,2);
min_index = zeros(1, no_signals);
cent_index = zeros(1, no_signals);
max_index = zeros(1, no_signals);
min_freq = zeros(1, no_signals);
cent_freq = zeros(1, no_signals);
max_freq = zeros(1, no_signals);

for ii = 1:no_signals;
	[max_val, cent_index(ii)] = max(abs(spec(:, ii)));
	spec(:, ii) = abs(spec(:, ii)) / max_val;
	indices = find(spec(:, ii) > min_value);
	min_index(ii) = min(indices);
	max_index(ii) = max(indices);
	min_freq(ii) = (min_index(ii) - 1) * freq_step;
	cent_freq(ii) = (cent_index(ii) - 1) * freq_step;
	max_freq(ii) = (max_index(ii) - 1) * freq_step;
end;

return;
