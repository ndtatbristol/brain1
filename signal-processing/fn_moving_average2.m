function s_av = fn_moving_average2(s, n, varargin)
%SUMMARY
%   Computes n-point moving average of supplied signal - effectively
%   providing a low-pass filter
%USAGE
%   function s_av = fn_moving_average(s, n [, pad])
%INPUTS
%   s - vector (or column matrix) of signals to average
%   n - number of points to use in moving average (which will be rounded up
%   to next odd number). Note n=1 does nothing!
%   pad[1] - extend s at both ends so result of averaging has same number
%   of rows as original s. If not padded result will have n-1 less rows.
%OUTPUTS
%   s_av - result of averaging

%NOTES
%   This version ignores NaNs in s (they are given zero weighting in
%   average calculation)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%set defaults
default_pad = 1;
if length(varargin) < 1
    pad = default_pad;
else
    pad = varargin{1};
end

%sort out n to be odd integer
n = ceil((n-1) / 2) * 2 + 1;

%extend s out if padding is requested
if pad
    s = [ones((n-1) / 2,1) * s(1,:); s; ones((n-1) / 2,1) * s(end,:)];
end
invalid_pts = isnan(s);
valid_pts = ~isnan(s);
s(find(invalid_pts)) = 0;
s_av = zeros(size(s,1) - n + 1, size(s,2));
tot = zeros(size(s,1) - n + 1, size(s,2));
for ii = 1:n;
    s_av = s_av + s(ii:end - n + ii, :);
    tot = tot + valid_pts(ii:end - n + ii, :);
end
s_av = s_av ./ tot;
return;
        