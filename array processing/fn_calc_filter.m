function filt = fn_calc_filter(t, centre_freq, half_bandwidth,varargin)
%SUMMARY
%   Calculates frequency Gaussian domain filter vector based on specifed
%   centre frequency and half bandwidth
%USAGE
%   filt = fn_calc_filter(t, centre_freq, half_bandwidth [, db_down, force_zero])
%INPUTS
%   t - filter will have length(t) points
%   centre_freq - centre frequency of filter
%   half_bandwidth - half bandwidth of filter
%   db_down[40] - number of dB below peak of window used to define the width.
%	force_zero[1] - if true, points in window outside defined width are
%	set to zero

%OUTPUTS
%   filt - filter vector of size length(t) by 1, the first element 
%   corresponds to freq = 0
n = length(t);
max_freq = 1 / (t(2) - t(1));
if (length(varargin)>0)
    if (length(varargin)>1)
        filt = fn_gaussian(n, centre_freq / max_freq, half_bandwidth / max_freq,varargin{1},varargin{2});
    else
        filt = fn_gaussian(n, centre_freq / max_freq, half_bandwidth / max_freq,varargin{1});
    end
else
    filt = fn_gaussian(n, centre_freq / max_freq, half_bandwidth / max_freq, 40, 1);
end
return;