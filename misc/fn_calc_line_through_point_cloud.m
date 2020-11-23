function [m, c] = fn_calc_line_through_point_cloud(x, y, varargin)
%SUMMARY
%   Performs weighted least squares through cloud of points defined by
%   (x,y) pairs. Optional arguments can force line through origin and add
%   weighting vector w.
%USAGE
%   [m, c] = fn_calc_line_through_point_cloud(x, y, [force_line_through_origin, w])
%INPUTS
%   x, y = vectors or matrices of point coordinates (must have same number
%   of elements)
%   Will be treated as one point cloud regardless of format
%   force_line_through_origin = 0 - force line to go through origin (c = 0)
%   if set to 1
%   w - vector or matrix of weights associated with each point in cloud.
%   Must have same number of elements as x and y.
%OUTPUTS
%   m - gradient of best fit line, y = m * x + c
%   c - intercept of best fit line, y = m * x + c
%--------------------------------------------------------------------------
x = x(:);
y = y(:);
if length(x) ~= length(y)
    error('Number of elements in x and y must be equal');
    return
end

if length(varargin) > 0
    force_line_through_origin = varargin{1};
else
    force_line_through_origin = 0;
end
if length(varargin) > 1
    w = varargin{2};
    w = w(:);
    if length(w) ~= length(x)
        error('Number of elements in x, y and w must be equal');
        return
    end
else
    w = ones(size(x));
end



if force_line_through_origin
    m = sum(y .* x .* w) / sum(x .^ 2 .* w);
    c = 0;
else
    M = [sum(x .^ 2 .* w), sum(x .* w); ...
        sum(x .* w), sum(w)];
    V = [sum(y .* x .* w); sum(y .* w)];
    tmp = M \ V;
    m = tmp(1);
    c = tmp(2);
end
end