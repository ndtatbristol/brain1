function [d, s] = fn_return_dimension_of_array(array)
% SUMMARY
%   Returns dimension of array (1D or 2D) (assuming array is in standard
%   coordinates (x, y in plane of array with x along axis for 1D array)
% INPUTS
%   array - array structure in usual NDT lab format
% OUTPUTS
%   d - numerical dimension (1 or 2)
%   s - string version ('1D' or '2D')
if any(array.el_yc)
    d = 2;
    s = '2D';
else
    d = 1;
    s = '1D';
end
end