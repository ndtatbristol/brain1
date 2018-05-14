function new_signal = fn_stetch_signal(old_signal, old_pad_pts, new_pad_pts)
%SUMMARY
%   Stretch or compress signal(s) by ratio new_pad_pts / old_pad_pts using
%   frequency domain technique
%USAGE
%   new_signal = fn_stetch_signal(old_signal, old_pad_pts, new_pad_pts, maintain_length)
%INPUTS
%   old_signal - vector (or column matrix) of signal(s) to stretch
%   old_pad_pts - number of pts to use in forward FFT (either scalar in
%   which case same value is applied to all signals or vector with length
%   equal to number of columns in old_signal to apply different values to
%   each signal)
%   new_pad_pts - number of pts to use in inverse FFT (either scalar or
%   vector as for old_par_pts)
%OUTPUTS
%   new_signal - stretched or compressed signals

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

old_pad_pts = round(old_pad_pts);
if length(old_pad_pts) == 1
    old_pad_pts = old_pad_pts * ones(1, size(old_signal, 2));
end;
new_pad_pts = round(new_pad_pts);
if length(new_pad_pts) == 1
    new_pad_pts = new_pad_pts * ones(1, size(old_signal, 2));
end;
sig_length = size(old_signal, 1);
new_signal = zeros(size(old_signal));
for ii = 1:size(old_signal, 2)
    spec = fft(old_signal(:, ii), old_pad_pts(ii));
    spec = spec(1:ceil(length(spec) / 2));
    temp = 2 * real(ifft(spec, new_pad_pts(ii))) * new_pad_pts(ii) / old_pad_pts(ii);
    if length(temp) >= sig_length
        new_signal(:, ii) = temp(1:sig_length);
    end;
    if length(temp) < length(old_signal)
        new_signal(:, ii) = [temp; zeros(sig_length - length(temp), 1)];
    end;
end;

return;