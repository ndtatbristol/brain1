function new_signal = fn_time_shift_signal(old_signal, time_step, time_shift);
%USAGE
%   new_signal = fn_time_shift_signal(old_signals, time_step, time_shifts);
%AUTHOR
%	Paul Wilcox (2004)
%SUMMARY
%	Applies time shifts to one or more time-domain signals
%OUTPUTS
%	new_signal - vector or matrix of time shifted signals (by column)
%INPUTS
%	old_signal - vector or matrix of original signals (by column)
%   time_step - time step between adjacent points in time signals
%   time_shift - the time shift to apply. If time_shifts is a scalar then
%   all the signals are shifted by the same amount. If it is a vector, then
%   it should have as many elements as there are columns in old_signals and
%   the corresponding time shift will be applied to each signal. Positive
%   values mean that the signal is advanced, negative values mean that the
%   signal is retarded.
%NOTES
%	Routine uses Fourier transforms and hence signals may wrap around when
%	shifted

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fft_pts = 2 ^ nextpow2(size(old_signal, 1));
spec = fft(old_signal, fft_pts);
spec = spec(1:fft_pts / 2, :);
freq = [0:fft_pts / 2 - 1]' / (time_step * fft_pts);
if length(time_shift) == 1
    time_shift = ones(1, size(old_signal, 2)) *  time_shift;
end;
spec = spec .* exp(2 * pi * i * (freq * time_shift));
new_signal = real(ifft(spec, fft_pts)) * 2;
new_signal = new_signal(1:size(old_signal,1), :);
return;