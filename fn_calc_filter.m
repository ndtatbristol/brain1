function filt = fn_calc_filter(t, centre_freq, half_bandwidth)
%SUMMARY
%   Calculates frequency Gaussian domain filter vector based on specifed
%   centre frequency and half bandwidth
%INPUTS
%   t - filter will have length(t) points
%   centre_freq - centre frequency of filter
%   half_bandwidth - half bandwidth of filter
%OUTPUTS
%   filt - filter vector of size length(t) by 1, the first element 
%   corresponds to freq = 0


n = length(t);
max_freq = 1 / (t(2) - t(1));
filt = fn_gaussian(n, centre_freq / max_freq, half_bandwidth / max_freq);
return;