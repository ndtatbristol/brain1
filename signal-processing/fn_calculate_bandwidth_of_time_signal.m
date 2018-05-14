function [min_freq, cent_freq, max_freq] = fn_calculate_bandwidth_of_time_signal(time, signal, varargin);
%USAGE
%	[min_freq, cent_freq, max_freq] = fn_calculate_bandwidth_of_time_signal(time, signal [, db_down]);
%SUMMARY
%	Calculates the bandwidth of time-domain signal(s). Only use if not
%	going to frequency domain anyway to avoid unnecessary FFTs. If in
%	frequency domain, use fn_calculate_bandwidth_of_spectrum instead.
%AUTHOR
%	Paul Wilcox (2007)
%INPUTS
%	time - vector representing time axis for signal(s)
%	signal - vector containing a single time signal, or matrix containigng
%	a time signal in each column, in which case, no. of rows must equal
%	length of time vector
%	db_down [40] - number of dB down from peak amplitude on which bandwidth is
%	based.
%OUTPUTS
%	min_freq, cent_freq and max_freq - the minimum, maximum and peak
%	frequencies. If signal is matrix, then these will be vectors with as
%	many elements as there are columns in signal.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
default_db_down = 40;
if length(varargin) < 1
	db_down = default_db_down;
else
	db_down = varargin{1};
end;
signal = squeeze(signal);
fft_pts = 2 ^ nextpow2(length(time));
time_step = abs(time(1) - time(2));
freq_step = 1 / (time_step * fft_pts);
spec = fft(signal, fft_pts);
spec = spec(1:fft_pts / 2, :);

[min_freq, cent_freq, max_freq, min_index, cent_index, max_index] = fn_calculate_bandwidth_of_spectrum(freq_step, spec, db_down);

% min_value = 10 ^ (-db_down / 20);
% 
% no_signals = size(spec,2);
% min_index = zeros(1, no_signals);
% cent_index = zeros(1, no_signals);
% max_index = zeros(1, no_signals);
% min_freq = zeros(1, no_signals);
% cent_freq = zeros(1, no_signals);
% max_freq = zeros(1, no_signals);
% 
% for ii = 1:no_signals;
% 	[max_val, cent_index(ii)] = max(abs(spec(:, ii)));
% 	spec(:, ii) = abs(spec(:, ii)) / max_val;
% 	indices = find(spec(:, ii) > min_value);
% 	min_index(ii) = min(indices);
% 	max_index(ii) = max(indices);
% 	min_freq(ii) = (min_index(ii) - 1) * freq_step;
% 	cent_freq(ii) = (cent_index(ii) - 1) * freq_step;
% 	max_freq(ii) = (max_index(ii) - 1) * freq_step;
% end;

return;
